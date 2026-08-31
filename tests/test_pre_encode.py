"""Regression tests for the pixel path's VAE-encode timing.

`vae.encode` reaches ComfyUI's `load_models_gpu(..., memory_required=...)`, which calls
`free_memory()` with an empty `keep_loaded` — so it will partially unload whatever is
already resident on the device. When the first encode landed *inside* the sampling
wrapper, that victim was the diffusion model the sampler was in the middle of using, and
nothing re-expands it (`sampler_helpers` loads once, before the loop). The rest of the run
then streamed weights from CPU on every step.

Native-grid sources are encoded at node-execution time, before the sampler loads
anything, because their encoding no longer depends on the target resolution. These
tests pin that timing and the no-crop, edge-padding-only geometry.

Self-contained: installs a minimal ComfyUI stub when the real one is not importable, so
it runs both inside a ComfyUI checkout and standalone.

Run standalone:  python tests/test_pre_encode.py
Or via pytest:   pytest tests/test_pre_encode.py
"""
import importlib.util
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))   # .../ComfyUI, when installed there

if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _install_comfy_stubs():
    """Only the surface the pack imports at module level. Skipped when real comfy is
    importable, so this never shadows a genuine ComfyUI checkout."""
    try:
        import comfy.patcher_extension  # noqa: F401
        return
    except Exception:                                            # noqa: BLE001
        pass

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    class WrapperExecutor:
        def __init__(self, original, class_obj, wrappers, idx):
            self.original, self.class_obj = original, class_obj
            self.wrappers, self.idx = list(wrappers), idx
            self.is_last = idx == len(wrappers)

        def execute(self, *args, **kwargs):
            if self.is_last:
                return self.original(*args, **kwargs)
            return self.wrappers[self.idx](self, *args, **kwargs)

    def add_wrapper_with_key(wrapper_type, key, wrapper, options):
        options.setdefault("wrappers", {}).setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    mods = {}
    for name in ("comfy", "comfy.patcher_extension", "comfy.utils", "comfy.ldm",
                 "comfy.ldm.common_dit", "comfy.ldm.flux", "comfy.ldm.flux.layers",
                 "comfy.ldm.flux.math", "comfy.ldm.modules", "comfy.ldm.modules.attention"):
        mods[name] = types.ModuleType(name)

    mods["comfy.patcher_extension"].WrappersMP = WrappersMP
    mods["comfy.patcher_extension"].WrapperExecutor = WrapperExecutor
    mods["comfy.patcher_extension"].add_wrapper_with_key = add_wrapper_with_key
    mods["comfy.utils"].common_upscale = lambda s, w, h, *a: s
    mods["comfy.ldm.common_dit"].pad_to_patch_size = lambda v, _p, **_k: v
    mods["comfy.ldm.flux.layers"].timestep_embedding = lambda t, d: torch.zeros(t.shape[0], d)
    mods["comfy.ldm.flux.math"].apply_rope = lambda q, k, _f: (q, k)
    mods["comfy.ldm.modules.attention"].optimized_attention_masked = lambda *a, **k: None
    mods["comfy.ldm.modules.attention"].attention_pytorch = lambda *a, **k: None

    for attr, parent, child in (("patcher_extension", "comfy", "comfy.patcher_extension"),
                                ("utils", "comfy", "comfy.utils"),
                                ("ldm", "comfy", "comfy.ldm"),
                                ("common_dit", "comfy.ldm", "comfy.ldm.common_dit"),
                                ("flux", "comfy.ldm", "comfy.ldm.flux"),
                                ("modules", "comfy.ldm", "comfy.ldm.modules"),
                                ("layers", "comfy.ldm.flux", "comfy.ldm.flux.layers"),
                                ("math", "comfy.ldm.flux", "comfy.ldm.flux.math"),
                                ("attention", "comfy.ldm.modules", "comfy.ldm.modules.attention")):
        setattr(mods[parent], attr, mods[child])
    sys.modules.update(mods)


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (the folder has a hyphen)."""
    _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location(
        "comfyui_krea2edit_pre_encode_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeVAE:
    """Records every encode so tests can assert on call timing and count."""

    def __init__(self):
        self.calls = []

    def encode(self, pixels):        # pixels: (B, H, W, C)
        self.calls.append(tuple(pixels.shape))
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 4, h // 8, w // 8)

    def decode(self, samples):       # samples: (B, C, H, W)
        b, _c, h, w = samples.shape
        return torch.zeros(b, h * 8, w * 8, 3)


class _FakeInner:
    def process_latent_in(self, x):
        return x


class _FakeModelPatcher:
    """Minimal stand-in for ModelPatcher — only what patch() touches."""

    def __init__(self):
        self.model = _FakeInner()
        self.model_options = {}

    def clone(self):
        c = _FakeModelPatcher()
        c.model = self.model      # share the inner model, like the real clone
        return c


def _latent(h=64, w=64, batch=1):
    return {"samples": torch.zeros(batch, 4, h, w)}


def _image(h=512, w=None, batch=1):
    return torch.zeros(batch, h, h if w is None else w, 3)


def _patch(mod, **kwargs):
    import comfy.patcher_extension as pe

    node = mod.Krea2EditModelPatch()
    source_latent = kwargs.pop("source_latent", _latent())
    (m,) = node.patch(_FakeModelPatcher(), source_latent, **kwargs)
    to = m.model_options["transformer_options"]
    return to["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0]


def _run_steps(wrapper, n=3, h=64, w=64):
    """Drive the wrapper the way a sampler would: same latent shape, n times."""
    import comfy.patcher_extension as pe

    for _ in range(n):
        executor = pe.WrapperExecutor(
            original=lambda *a, **k: None, class_obj=object(), wrappers=[wrapper], idx=0,
        )
        executor.execute(torch.zeros(1, 4, h, w), None, None, None, {})


def _stub_forward(mod, seen):
    mod.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: seen.append(src)


def test_pixel_source_encodes_before_sampling():
    """Native-grid source encoding completes during patch(), before sampling begins."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), target_latent=_latent())

    assert len(vae.calls) == 1, "source should be encoded during patch(), not later"

    _run_steps(wrapper)
    assert len(vae.calls) == 1, "sampling must reuse the pre-encode, never re-encode"
    assert seen[0].shape == (1, 4, 64, 64)


def test_target_latent_is_not_needed_for_preencode():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image())

    assert len(vae.calls) == 1, "native-grid source must encode without target_latent"

    _run_steps(wrapper)
    assert len(vae.calls) == 1, "sampling must reuse the native-grid source"


def test_both_references_are_pre_encoded():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), source_image_b=_image(),
                     target_latent=_latent())

    assert len(vae.calls) == 2, "scene and subject refs both pre-encoded"

    _run_steps(wrapper)
    assert len(vae.calls) == 2
    assert isinstance(seen[0], list) and len(seen[0]) == 2


def test_reference_image_batches_expand_to_multiple_refs_in_order():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(
        mod,
        vae=vae,
        source_image=_image(batch=2),
        source_image_b=_image(batch=1),
        reference_images=_image(batch=3),
    )

    assert len(vae.calls) == 6
    assert all(call == (1, 512, 512, 3) for call in vae.calls)
    _run_steps(wrapper)
    assert isinstance(seen[0], list) and len(seen[0]) == 6


def test_reference_latent_batches_expand_to_multiple_refs_in_order():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    wrapper = _patch(
        mod,
        source_latent=_latent(batch=2),
        source_latent_b=_latent(batch=1),
        reference_latents=_latent(batch=3),
    )

    _run_steps(wrapper)
    assert isinstance(seen[0], list) and len(seen[0]) == 6


def test_target_latent_does_not_change_native_source_geometry():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(), target_latent=_latent(32, 32))

    assert len(vae.calls) == 1

    _run_steps(wrapper, n=2, h=64, w=64)
    assert len(vae.calls) == 1, "target resolution must not trigger a re-encode"
    assert seen[0].shape == (1, 4, 64, 64)


def test_pixel_source_keeps_its_native_grid_with_only_edge_padding():
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    vae = _FakeVAE()
    wrapper = _patch(mod, vae=vae, source_image=_image(513, 777), target_latent=_latent(32, 32))

    # 513x777 is padded (not cropped/resized) to the next 16-pixel lattice.
    assert vae.calls == [(1, 528, 784, 3)]
    _run_steps(wrapper)
    assert seen[0].shape == (1, 4, 66, 98)


def test_target_latent_without_pixel_path_is_inert():
    """No vae/source_image means the latent path — target_latent must change nothing."""
    mod = _load_pack()
    seen = []
    _stub_forward(mod, seen)
    wrapper = _patch(mod, target_latent=_latent())
    _run_steps(wrapper)
    assert seen[0].shape == (1, 4, 64, 64)


def test_arbitrary_size_canvas_round_trips_exact_pixel_size():
    """The sampling canvas may be any HxW; only its VAE alignment pad is removed."""
    mod = _load_pack()
    vae = _FakeVAE()

    latent, width, height = mod.Krea2EditEmptyLatent().make(vae, 513, 777)

    # The VAE works on the /8 ceiling grid.  The requested geometry is preserved
    # separately so decode can remove only the bottom/right alignment pixels.
    assert vae.calls == [(1, 784, 520, 3)]
    assert latent["samples"].shape == (1, 4, 98, 65)

    (image,) = mod.Krea2EditVAEDecode().decode(vae, latent, width, height)
    assert image.shape == (1, 777, 513, 3)


def test_grounded_encode_keeps_image_grounded_text_but_removes_visual_positions():
    """The VLM must run on image positions, but the returned DiT context must not
    retain those positions (or their matching attention-mask entries)."""
    mod = _load_pack()

    class _FakeVLM:
        def __init__(self):
            self.was_called = False

        def build_image_inputs(self, *_args, **_kwargs):
            self.was_called = True
            # Full VLM sequence: two stripped-prefix tokens, then a 4-token DiT
            # suffix where positions 1 and 3 are visual patches.
            return None, torch.tensor([[False, True, False, True, False, True]]), None

    class _FakeClip:
        def __init__(self):
            self.vlm = _FakeVLM()
            self.cond_stage_model = type("Stage", (), {
                "transformer": type("ClipModel", (), {"transformer": self.vlm})()
            })()

        def encode_from_tokens_scheduled(self, _tokens):
            self.vlm.build_image_inputs(None, [])
            cond = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
            return [[cond, {"attention_mask": torch.tensor([[1, 1, 0, 1]])}]]

    clip = _FakeClip()
    result = mod._encode_grounded_text_only(clip, object())

    cond, options = result[0]
    assert clip.vlm.was_called, "the image must still be encoded by Qwen3-VL"
    assert cond.tolist() == [[[0.0, 1.0], [4.0, 5.0]]]
    assert options["attention_mask"].tolist() == [[1, 0]]
    assert "build_image_inputs" not in clip.vlm.__dict__, "temporary capture hook must be removed"


def test_grounded_encode_expands_all_reference_image_batches():
    mod = _load_pack()
    seen = {}

    class _FakeClip:
        def tokenize(self, prompt, images, llama_template):
            seen["prompt"] = prompt
            seen["images"] = images
            seen["template"] = llama_template
            return object()

    original = mod._encode_grounded_text_only
    mod._encode_grounded_text_only = lambda _clip, _tokens: "grounded"
    try:
        result = mod.Krea2EditGroundedEncode().encode(
            _FakeClip(),
            "combine the references",
            image=_image(batch=2),
            image_b=_image(batch=1),
            reference_images=_image(batch=3),
        )
    finally:
        mod._encode_grounded_text_only = original

    assert result == ("grounded",)
    assert len(seen["images"]) == 6
    assert seen["template"].count("<|vision_start|>") == 6


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:                               # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
