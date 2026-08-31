"""Regression test: the DIFFUSION_MODEL wrapper must accept the exact positional
call shape that comfy/ldm/krea2/model.py uses.

Original test contributed by @akashzeno (GitHub PR #9, thank you). The second test
here is adapted to our fix strategy: rather than mirroring forward's signature
name-for-name, our wrapper absorbs trailing positionals with ``*wargs`` and picks
``transformer_options`` as the trailing dict, so it tolerates future argument
insertions too. So test 2 checks the *behavior* (the right dict is picked up)
instead of the signature shape.

Upstream commit c9602625 inserted a new ``ref_latents`` positional argument between
``attention_mask`` and ``transformer_options`` in SingleStreamDiT.forward:

    .execute(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

WrapperExecutor forwards these positionally as ``wrapper(self, x, timesteps, context,
attention_mask, ref_latents, transformer_options)`` -> 7 positional args. A wrapper that
predates the change only has 6 slots and dies with:

    TypeError: wrapper() takes from 4 to 6 positional arguments but 7 were given

These tests reproduce that call path with a stubbed forward so they stay fast and
model-free, and guard against the signature drifting out of sync again.

Run standalone:  python tests/test_wrapper_signature.py
Or via pytest:   pytest tests/test_wrapper_signature.py
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)                       # .../custom_nodes/comfyui-krea2edit
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))  # .../ComfyUI

# ComfyUI root must be importable so the pack's ``import comfy.*`` lines resolve.
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (folder has a hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "comfyui_krea2edit_under_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeInner:
    def process_latent_in(self, x):
        return x


class _FakeModelPatcher:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher, only what patch() touches."""
    def __init__(self):
        self.model = _FakeInner()
        self.model_options = {}

    def clone(self):
        c = _FakeModelPatcher()
        c.model = self.model  # share inner model, like the real clone
        return c


class _FakeSamples:
    """Minimal 4D latent batch for wrapper-only tests (no tensor dependency)."""
    ndim = 4
    shape = (1, 4, 1, 1)

    def __getitem__(self, _index):
        return self


def _get_wrapper(mod):
    import comfy.patcher_extension as pe

    node = mod.Krea2EditModelPatch()
    (m,) = node.patch(_FakeModelPatcher(), {"samples": _FakeSamples()})
    to = m.model_options["transformer_options"]
    return to["wrappers"][pe.WrappersMP.DIFFUSION_MODEL]["krea2_edit"][0]


def test_wrapper_accepts_upstream_positional_call():
    """The crash path: new-style 6-positional call (with ref_latents) must not raise."""
    import comfy.patcher_extension as pe

    mod = _load_pack()
    wrapper = _get_wrapper(mod)

    sentinel = object()
    mod.krea2_edit_forward = lambda *a, **k: sentinel

    executor = pe.WrapperExecutor(
        original=lambda *a, **k: None,
        class_obj=object(),   # stands in for the SingleStreamDiT instance
        wrappers=[wrapper],
        idx=0,
    )

    result = executor.execute(
        None,   # x
        None,   # timesteps
        None,   # context
        None,   # attention_mask
        None,   # ref_latents  <-- the argument added by c9602625
        {},     # transformer_options
    )
    assert result is sentinel


def test_transformer_options_picked_up_under_both_signatures():
    """Behavioral guard: transformer_options must bind to the trailing dict, never to
    ref_latents, under both the old (5-arg) and new (6-arg) positional shapes."""
    import comfy.patcher_extension as pe

    mod = _load_pack()

    captured = {}
    mod.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: captured.__setitem__("to", to)

    marker_old = {"tag": "old"}
    marker_new = {"tag": "new"}

    for shape, marker in (("old", marker_old), ("new", marker_new)):
        wrapper = _get_wrapper(mod)
        executor = pe.WrapperExecutor(
            original=lambda *a, **k: None, class_obj=object(),
            wrappers=[wrapper], idx=0,
        )
        if shape == "old":
            executor.execute(None, None, None, None, marker)              # ...attention_mask, t_opts
        else:
            executor.execute(None, None, None, None, ["ref"], marker)     # ...ref_latents, t_opts
        assert captured["to"] is marker, f"{shape}: got {captured['to']}, expected the trailing dict"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
