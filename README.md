# ComfyUI-Krea2Edit

Instruction-based image editing for **Krea 2** in ComfyUI — the node pack that powers
the **Krea 2 Identity Edit** LoRA. Turns Krea 2 (Raw or Turbo) into an image editor with dual
conditioning: the source image is injected both as VAE latent tokens (appearance) and
into the Qwen3-VL text encoder (semantic grounding). The VLM's visual-patch hidden
states are then removed, so only image-grounded language-token states enter the DiT.
Source latent tokens receive the clean endpoint `t=0` AdaLN modulation, while target
tokens receive the sampler's current timestep; attention still runs over both blocks.
Each image has an independent RoPE grid: references keep their native aspect ratio and
begin at `(h=0, w=0)` in their own frame. The node never crops or resizes a reference;
it adds only bottom/right edge padding needed for the 16-pixel VAE/DiT lattice.

☕ **[Support on Ko-fi](https://ko-fi.com/conradlocke)** — all tips go straight to GPU compute for future versions.

🧰 **Training code is public:** [krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer) — the ai-toolkit extension these LoRAs were trained with, geometry-matched to the nodes, with measured consumer-GPU VRAM requirements.

## Model versions

This native-grid fork requires a LoRA trained with the paired native-grid trainer. It
is not geometry-compatible with released v1/v1.1/v1.2 Identity Edit LoRAs, which used
target-fitted or cropped references.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lbouaraba/comfyui-krea2edit
# restart ComfyUI
```

Requirements: a ComfyUI version with native Krea 2 support, the Krea 2 model
(Raw or Turbo), the Qwen3-VL 4B text encoder used by Krea 2, and the Krea 2 Identity Edit
LoRA (`krea2_identity_edit_v1_2.safetensors`). No extra Python dependencies.

## Nodes

### `Krea2EditModelPatch`
Wraps the diffusion model so every VAE-encoded reference is added as clean
in-context tokens (RoPE frames 1…N, `t=0` block modulation). It is internally placed
after the target block only to use Krea2's native clean-time routing; attention is
non-causal and the RoPE positions are unchanged. Inputs:
- `model` — Krea 2 (LoRA already applied)
- `source_latent` / `source_latent_b` — legacy primary/additional references. A
  latent batch is split so each batch item becomes one reference.
- `reference_latents` — additional references as an N-sample latent batch.
- `vae` + `source_image` *(optional, recommended)* — VAE-encodes the original source
  image in pixel space, preserving its native aspect ratio and resolution. It is only
  edge-padded to the 16-pixel patch lattice; no crop or resize is performed.
- `source_image_b` — legacy additional image reference.
- `reference_images` — additional references as an N-image batch. Each image becomes
  a separate native-grid reference; its order must match the text node. A standard
  ComfyUI IMAGE batch has one shared image size; list-producing nodes may also pass a
  list of differently sized IMAGE batches.
- `target_latent` *(optional, legacy compatibility)* — no longer needed: native-grid
  source encoding does not depend on output resolution.
- `ref_boost` *(default 1.0)* — reference-fidelity dial; >1 pulls harder toward the
  reference's appearance, <1 loosens. `ref_boost_a` is the same dial for the scene ref in two-ref edits.

### `Krea2EditGroundedEncode`
Image-grounded instruction encoding — the text encoder *sees* the image while
reading your instruction. Its visual-patch hidden states are removed before the
conditioning reaches the DiT, while the remaining language-token states retain the
VLM's image grounding. Inputs:
- `clip` — the Krea 2 CLIP (Qwen3-VL, loaded with `type: krea2`)
- `prompt` — the edit instruction ("recolor the car to matte black")
- `image` / `image_b` — primary and legacy additional references
- `reference_images` — additional references as an N-image batch. Connect the same
  images in exactly the same order as the model-patch node.
- `grounding_px` — grounding resolution (default 768; v1.2 trained range
  384–768, and 1024+ often still works nicely). This is a quality dial: lower =
  stronger edit adherence, higher = stronger identity/likeness. Try 1024 for
  people, 512 for stubborn scene changes. (v1's trained range was 512–1536.)

**Both nodes are required.** With a stock `CLIPTextEncode` the model never sees the
image semantically and quality drops sharply, especially for scene-referential
instructions ("the man on the left").

## Minimal wiring

```
LoadImage ─┬─ VAEEncode ── Krea2EditModelPatch.source_latent
           └─ Krea2EditGroundedEncode.image     (+ your prompt)
UNETLoader ── LoraLoaderModelOnly (krea2_identity_edit_v1_2 @1.0) ── Krea2EditModelPatch.model
Krea2EditModelPatch ── KSampler.model
Krea2EditGroundedEncode ── KSampler.positive
Krea2EditGroundedEncode (empty prompt, same image) ── KSampler.negative
EmptySD3LatentImage ─┬─ KSampler.latent_image
```

Example workflow in `workflows/`: `krea2_identity_edit.json` — single-image editor by
default; enable group 2 (toggle its Bypass off) for two-image person-into-scene edits.

## Usage notes (read these — they matter)

1. **Aspect ratio.** Source and output may use different aspect ratios. Each reference
   has its own `(frame, h, w)` RoPE grid and is never fitted to the output grid. Larger
   source images create more reference tokens, so keep their native resolution within
   your VRAM budget.
2. **Turbo, 8 steps, CFG 1** is the fast path (~1 min at 2MP) and works for most
   edits: recolor, add/insert, attribute changes, restyles, scene translation.
3. **Removals and other "delete salient content" edits need real guidance:**
   use the **Raw** model at **CFG 3, ~20 steps**. Distilled Turbo at CFG 1 will
   usually re-render the subject instead of removing it.
4. At CFG > 1, ground the negative too: a second `Krea2EditGroundedEncode` with an
   empty prompt and the same image (this is the trained unconditional).
5. **Multi-reference order matters.** A source/image batch is split in batch order;
   then `source_*_b` / `image_b`; then `reference_latents` / `reference_images`.
   Feed the exact same ordered image set to `Krea2EditGroundedEncode`. Each reference
   gets frame `1…N`; `ref_boost` and `ref_boost_mask` apply to the final reference,
   while `ref_boost_a` applies to all earlier references.
6. **Generate at ≤2MP.** Above the trained range, source content can bleed into
   the output or subjects duplicate.
7. **Two distinct people:** place both references in a single pass (scene/subject A on
   the main inputs, subject B on the `_b` inputs) rather than adding them one at a time —
   simultaneous placement is currently more reliable than chaining separate edits. Face
   separation is still imperfect and a focus for future versions.

## Pixel path and VRAM

The pixel path VAE-encodes each source once at node-execution time, before sampling,
on that source's native pixel grid. It needs no target resolution and therefore no
longer needs `target_latent`. The source is padded only on its bottom/right edge to a
multiple of 16 pixels; it is not cropped or resampled.

More references increase both DiT sequence length and VLM work. Keep the aggregate
reference resolution within your VRAM/context budget.

The console tells you which path you got:

```
[krea2edit] native source VAE encode: (513, 777) -> (528, 784) px (edge padding only)
```

You may instead supply `source_latent` directly. Its native latent grid is used as-is,
apart from DiT patch alignment padding.

## License / credits

Nodes: Apache-2.0. The **Krea 2 Identity Edit** weights ship separately under the
Krea 2 Community License Agreement (see the model card, `LICENSE.pdf`, and `NOTICE`
in the weights repo).
Built on Krea 2 by Krea AI; text encoder Qwen3-VL (Alibaba).

## Contributors & thanks

This is a solo project, made a lot better by the community. Thank you to:

- **[stablellama](https://huggingface.co/stablellama)** — the MIT-licensed head/face/eye/person
  swap dataset behind those capabilities in v1.2.
- **[CeciliaXCIX](https://huggingface.co/CeciliaXCIX)** — tireless, high-quality community
  support in the discussions.
- **[akashzeno](https://github.com/akashzeno)** — node engineering: diagnosing the ComfyUI
  compatibility break and contributing the regression test.
- **[SubtleShader](https://huggingface.co/SubtleShader)** — testing the training code and
  consumer-GPU feedback.
- **Mark ([sogni.ai](https://sogni.ai))** — support, a GPU-fund donation, and getting the word out.
- **[ethanfel](https://github.com/ethanfel)** — root-caused the pixel path's VRAM interaction with
  the sampler and contributed the `target_latent` pre-encode (#15), tests included.

Want to help? Contributions of training data and node/code work are welcome, see the discussions.

## Scope and responsible use

Krea 2 Identity Edit is an identity-preserving character restaging model, trained only on
SFW data. It is not trained on any NSFW concepts, and I have no plans to add or support NSFW
data in current or future versions.

I do not endorse or support using this model to produce non-consensual, harmful, or sexual
imagery of real people, including deepfakes. Please use it responsibly and respect the
consent and likeness of anyone you depict.
