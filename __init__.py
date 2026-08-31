"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent is a block of clean tokens, distinguished from the
(noisy) target by the 3-axis RoPE frame index (source=1, target=0, h/w aligned) and by
its `t=0` block modulation. This node uses `[text | target(frame=0) | source(frame=1)]`
internally so native Krea2 can route the source suffix to `t=0`; self-attention is
non-causal and the RoPE IDs are unchanged. Only target tokens are returned.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditModelPatch(model, source_latent) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text: NATIVE krea2 CLIP + CLIPTextEncode.
"""
import math
import threading

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


# ``Krea2TEModel`` lives in ComfyUI, outside this node pack. During an image-grounded
# encode it expands each ``<|image_pad|>`` placeholder into a variable-length visual
# span, so the span cannot be reconstructed reliably from the prompt string alone.
# The lock scopes the small, temporary hook used to capture that authoritative mask.
_VLM_MASK_CAPTURE_LOCK = threading.RLock()


def _remove_visual_tokens_from_conditioning(conditioning, visual_masks):
    """Drop image-patch positions from ComfyUI Krea2 conditioning.

    Qwen3-VL has already processed the image when this runs: the retained language
    token states can therefore carry image-grounded semantics. Only positions marked
    by Qwen3-VL itself as visual patches are removed before the conditioning enters
    the DiT. The companion attention mask must be filtered identically.
    """
    if len(conditioning) != len(visual_masks):
        raise RuntimeError(
            "krea2edit: captured VLM visual-mask count does not match the number of "
            "ComfyUI conditioning segments; refusing to pass unfiltered visual tokens to the DiT."
        )

    filtered = []
    for (cond, options), visual_mask in zip(conditioning, visual_masks):
        if visual_mask is None:
            raise RuntimeError(
                "krea2edit: Qwen3-VL did not report visual positions for an image-grounded "
                "encode; refusing to pass unfiltered visual tokens to the DiT."
            )
        if cond.ndim != 3 or visual_mask.ndim != 2:
            raise RuntimeError(
                "krea2edit: unexpected conditioning or visual-mask rank while filtering VLM tokens."
            )

        # ComfyUI's Krea2 encoder removes its system/user prefix before returning
        # ``cond``. Align the full Qwen3-VL mask to that returned suffix.
        offset = visual_mask.shape[1] - cond.shape[1]
        if offset < 0:
            raise RuntimeError(
                "krea2edit: Qwen3-VL visual mask is shorter than the returned conditioning."
            )
        visual_mask = visual_mask[:, offset:]
        if visual_mask.shape != cond.shape[:2]:
            raise RuntimeError(
                "krea2edit: visual mask does not align with the returned Krea2 conditioning."
            )
        # A Comfy conditioning tensor has one shared sequence layout. Per-batch masks
        # that differ would require ragged conditioning, which DiT cannot represent.
        if not bool((visual_mask == visual_mask[:1]).all()):
            raise RuntimeError(
                "krea2edit: batch items have different visual-token layouts; cannot build "
                "a single text-only DiT conditioning tensor."
            )

        keep = ~visual_mask[0]
        next_options = dict(options)
        attention_mask = next_options.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.shape != cond.shape[:2]:
                raise RuntimeError(
                    "krea2edit: attention mask does not align with Krea2 conditioning."
                )
            next_options["attention_mask"] = attention_mask[:, keep]
        filtered.append([cond[:, keep, :], next_options])
    return filtered


def _encode_grounded_text_only(clip, tokens):
    """Encode through Qwen3-VL, then remove its visual states from DiT context.

    The hook is around the *real* ComfyUI encode call (not a second preprocessing
    pass), so it captures the exact dynamic visual span produced by the VLM.
    """
    try:
        vlm = clip.cond_stage_model.transformer.transformer
        original_build_image_inputs = vlm.build_image_inputs
    except AttributeError as exc:
        raise RuntimeError(
            "krea2edit: this node requires ComfyUI's native Qwen3-VL/Krea2 text encoder "
            "with build_image_inputs support."
        ) from exc

    visual_masks = []

    def capture_visual_mask(*args, **kwargs):
        result = original_build_image_inputs(*args, **kwargs)
        visual_masks.append(result[1])
        return result

    with _VLM_MASK_CAPTURE_LOCK:
        # ``build_image_inputs`` is normally resolved from Qwen3VL's class. Restore
        # that descriptor in ``finally`` even when ComfyUI's encode raises.
        vlm.build_image_inputs = capture_visual_mask
        try:
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            del vlm.build_image_inputs

    return _remove_visual_tokens_from_conditioning(conditioning, visual_masks)


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _encode_native_source_image(image, vae, cache, key):
    """VAE-encode a source on its own grid, without crop or resize.

    The VAE is /8 and Krea's DiT patch is 2x2 latent cells, so only bottom/right
    replicate padding to a 16-pixel lattice is required. It preserves every input
    pixel and lets the reference start its independent RoPE grid at (h=0, w=0).
    """
    if key in cache:
        return cache[key]
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    pad_h, pad_w = (-img.shape[-2]) % 16, (-img.shape[-1]) % 16
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
    pixels = img.movedim(1, -1)[..., :3].clamp(0, 1)
    print(f"[krea2edit] native source VAE encode: {tuple(image.shape[-3:-1])} -> "
          f"{tuple(pixels.shape[-3:-1])} px (edge padding only)", flush=True)
    cache[key] = vae.encode(pixels)
    return cache[key]


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
    """Additive attention-logit bias on the [text | target | refs...] sequence.

    boosts: per-ref factor on target->ref attention, aligned with the source blocks
    (last entry = last ref = the subject by workflow convention). Equivalent to
    multiplying those keys' post-softmax attention weight before renormalization.
    boost_mask (ComfyUI MASK, ref-image pixel space) restricts the LAST ref's boost
    to a region (e.g. the face).
    """
    nsrc = len(slens)
    # Reference tokens live after target tokens so ComfyUI's native Krea2 block can
    # efficiently apply a separate t=0 modulation to the contiguous suffix.
    target_start = txtlen
    target_end = target_start + tgtlen
    offs = [target_end]
    for sl in slens:
        offs.append(offs[-1] + sl)
    L = offs[-1]
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, target_start:target_end, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None):
    """Krea2 SingleStreamDiT._forward with clean-time source token blocks.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D — or a LIST of them
                  (multi-ref: [scene, subject], frames 1..N). Their block-level
                  time modulation is explicitly t=0, unlike noisy target tokens.
    context     : (B, seq, txtlayers*txtdim) — the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # Every source keeps its own latent grid.  We only pad its bottom/right edge to
    # the DiT patch lattice; no target-dependent crop, resize, or coordinate offset.
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)
    t_clean = m.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), m.tdim).unsqueeze(1).to(tgt_img.dtype)
    )
    # ComfyUI's native Krea2 block accepts a 2B modulation stack and applies its
    # second half to the suffix selected by ``timestep_zero_index``. This avoids a
    # B x sequence x 6D allocation and preserves full ref<->target self-attention.
    tvec = torch.cat((tvec, m.tproj(t_clean)), dim=0)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    # Put refs in a contiguous suffix. Self-attention has no causal ordering, and
    # RoPE position IDs remain unchanged, so this only enables native t=0 routing.
    combined = torch.cat([context, tgt_img] + src_imgs, dim=1)  # [text | target | refs...]
    timestep_zero_index = txtlen + tgtlen

    device = combined.device
    ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + [_imgids(bs, 0, h_, w_, device)]                                    # target frame=0
        + ref_ids,
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    if ref_boost != 1.0 or ref_boost_a != 1.0:
        # last ref = subject (single-ref: the only ref); earlier refs (scene) get ref_boost_a
        boosts = [ref_boost_a] * (len(src_imgs) - 1) + [ref_boost]
        attn_bias = _ref_attn_bias(boosts, ref_boost_mask, txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        try:
            combined = block(
                combined,
                tvec,
                freqs,
                attn_bias,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options,
            )
        except TypeError as exc:
            raise RuntimeError(
                "krea2edit: separate reference t=0 conditioning requires a current "
                "ComfyUI Krea2 implementation with SingleStreamBlock.timestep_zero_index. "
                "Update ComfyUI and restart it."
            ) from exc

    # LastLayer has no cross-token interaction. Run it only on target states with
    # target t, because reference outputs are intentionally discarded.
    out = m.last(combined[:, txtlen:txtlen + tgtlen], t)
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "source_latent": ("LATENT",),
        }, "optional": {
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo) for multi-ref LoRAs -> RoPE frame=2, training-matched order: scene first, subject second"}),
            "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                     "tooltip": "reference-fidelity dial: multiplies target->reference attention. Applies to the LAST ref (= the subject in two-ref workflows, the only ref in single-ref). 1.0 = off, >1 pulls harder toward the reference's appearance, <1 loosens. Optimal value is model-specific"}),
            "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "same dial for the FIRST ref (= the scene in two-ref workflows). No effect in single-ref workflows. 1.0 = off"}),
            "ref_boost_mask": ("MASK", {"tooltip": "optional region on the (last) reference to boost, e.g. the face; empty = whole reference"}),
            "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image: VAE-encodes the source in pixel space on its own native grid (no crop or resize)"}),
            "source_image": ("IMAGE", {"tooltip": "source as IMAGE (with vae connected): preserves its native pixel grid; only patch-alignment edge padding is added"}),
            "source_image_b": ("IMAGE", {"tooltip": "2nd reference as IMAGE (with vae)"}),
            # Retained for compatibility with existing workflows. Native reference
            # encoding no longer consumes it.
            "target_latent": ("LATENT", {"tooltip": "Legacy compatibility input. Native-grid source encoding no longer depends on target resolution, so this can be left disconnected."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Adds Krea2Edit source preservation (frame=1, clean t=0 source tokens) to a Krea2 model."

    def patch(self, model, source_latent, source_latent_b=None, ref_boost=1.0, ref_boost_a=1.0,
              ref_boost_mask=None, vae=None, source_image=None,
              source_image_b=None, target_latent=None, **_future):
        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). The workflow is newer than the "
                  f"installed node pack. Update comfyui-krea2edit (Manager -> Update, or git "
                  f"pull) and restart ComfyUI. Continuing without them.", flush=True)
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src_samples = model.model.process_latent_in(source_latent["samples"])
        if source_latent_b is not None:
            src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        px_cache = {}   # pixel-path encoded sources, one cache entry per reference
        mm = model.model  # for process_latent_in on the pixel path

        # Native-grid sources do not depend on output resolution, so always encode
        # outside the sampling window. This also makes target_latent unnecessary;
        # preserve that input only so older workflows retain their socket layout.
        if vae is not None and source_image is not None:
            src_samples = mm.process_latent_in(
                _encode_native_source_image(source_image, vae, px_cache, "a")
            )
            if source_image_b is not None:
                src_samples = [
                    src_samples,
                    mm.process_latent_in(
                        _encode_native_source_image(source_image_b, vae, px_cache, "b")
                    ),
                ]

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI signature drift (2026-07-19, commit c9602625 adds ref_latents):
            #   old: execute(x, t, ctx, attention_mask, transformer_options)
            #   new: execute(x, t, ctx, attention_mask, ref_latents, transformer_options)
            # Accept both: transformer_options is the trailing dict; any native
            # ref_latents are ignored (this patch supplies its own source path).
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            dm = executor.class_obj  # the SingleStreamDiT instance
            v = krea2_edit_forward(dm, x, timesteps, context, src_samples, transformer_options,
                                   ref_boost=ref_boost, ref_boost_a=ref_boost_a,
                                   ref_boost_mask=ref_boost_mask)
            return v

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (m,)


class Krea2EditGroundedEncode:
    """Image-grounded instruction encode — the SEMANTIC path of krea2_edit.

    The VLM encodes the instruction TOGETHER with the source image (user turn =
    <vision tokens: source> + instruction) and taps 12 layers. Its visual-patch
    hidden states are removed before DiT conditioning, leaving only image-grounded
    language-token states in the DiT text stream.
    Stock CLIPTextEncode is text-only, so inference was running with the grounding
    half of the recipe missing (the VAE source tokens carry appearance; THIS carries
    scene semantics: "the man on the left", "the sign in the back").

    Requires a qwen3vl TE checkpoint WITH the vision tower (all local ones have it).
    grounding_px caps the longest side fed to the VLM — the 2026-07-02 LoRA trained
    with 384-768px jitter, so 640-768 is in-distribution; 0 = native res (the jitter
    training makes that tolerable too). For CFG, ground the NEGATIVE too: second node,
    empty prompt, same image (matches training's unconditional).
    """
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )

    KREA2_EDIT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def _template(cls, nimg, system_prompt=""):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + vis + "{}<|im_end|>\n<|im_start|>assistant\n")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (subject) for multi-ref LoRAs; vision blocks in training order: scene, subject"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to Qwen3-VL; 0 = native"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "advanced (optional): override the grounding system prompt (empty = training default). Steers what the vision encoder attends to, e.g. facial identity detail."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Encodes an instruction with Qwen3-VL image grounding, then sends only its language-token states to the DiT."

    KREA2_EDIT_TEMPLATE_2REF = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768, system_prompt="", **_future):
        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). Update comfyui-krea2edit and restart "
                  f"ComfyUI. Continuing without them.", flush=True)
        if image is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)
        imgs = [self._prep(image, grounding_px)]
        if image_b is not None:
            imgs.append(self._prep(image_b, grounding_px))
        template = self._template(len(imgs), system_prompt)
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return (_encode_grounded_text_only(clip, tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatch": "Krea2 Edit (source patch)",
    "Krea2EditGroundedEncode": "Krea2 Edit (grounded encode)",
}


def _pack_version():
    # single source of truth = pyproject.toml, so this never drifts from the release
    try:
        import os, re
        p = os.path.join(os.path.dirname(__file__), "pyproject.toml")
        with open(p) as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


print(f"[krea2edit] nodes v{_pack_version()} loaded", flush=True)
