# ComfyUI Krea2Edit — Independent Multi-Reference Architecture

This node pack is the inference counterpart of
[chinoll/krea2edit-trainer](https://github.com/chinoll/krea2edit-trainer). It implements
a **new** Krea 2 edit-conditioning architecture with independent reference grids,
multiple references, image-grounded text, and clean-reference `t=0` modulation.

It is not a compatibility wrapper for released target-fitted/cropped Identity Edit
LoRAs. Use a LoRA trained with the paired ragged multi-reference trainer.

Upstream references:

- Original trainer: [lbouaraba/krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)
- Training framework: [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- Node host: [ComfyUI](https://github.com/Comfy-Org/ComfyUI)

## Inference contract

```text
references + instruction
        ├─ Qwen3-VL → discard visual hidden states → grounded language context
        └─ VAE → clean reference latent blocks

[text | target(frame=0, sampled t) | refs(frame=1..N, t=0)] → Krea 2 DiT
```

The node uses the `[text | target | refs]` order internally so ComfyUI's native Krea 2
blocks can apply `t=0` modulation to the contiguous reference suffix. Attention is
non-causal, so every reference still attends to the target and text. Each reference
has an independent RoPE grid starting at `(h=0, w=0)`; references are never cropped,
target-fitted, or position-offset.

The VLM sees every reference together with the instruction, but its visual-token
hidden states are removed before DiT conditioning. The DiT therefore receives
image-grounded language tokens plus clean VAE-latent appearance tokens.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/chinoll/comfyui-krea2edit
```

Restart ComfyUI. You need a current ComfyUI build with native Krea 2 support, a Krea
2 model, and the Qwen3-VL text encoder used by Krea 2. No extra Python package is
required by this node pack.

## Nodes

### `Krea2EditModelPatch`

Adds clean reference latent token blocks to the Krea 2 model.

- `source_image`, `source_image_b`, `reference_images`: image references. Connect a
  VAE; every image is VAE-encoded independently and becomes a separate RoPE frame.
- `source_latent`, `source_latent_b`, `reference_latents`: direct latent references.
  A latent batch is interpreted as an ordered reference list, not sampler batching.
- `ref_boost` and `ref_boost_a`: optional target-to-reference attention strength for
  the last and earlier references respectively.

Reference H/W is aligned to the nearest 16px VAE/DiT grid. It is not cropped or
limited by a node-side pixel budget.

### `Krea2EditGroundedEncode`

Builds Qwen3-VL image-grounded text conditioning.

- Connect the same references, in the same order, as `Krea2EditModelPatch`.
- `grounding_px` (default `768`) is the separate VLM longest-side cap. It applies
  only to Qwen3-VL grounding and controls VLM cost/detail.

Use this node for both positive and negative conditioning. Stock text-only Krea 2
encoding omits image grounding and does not match this architecture.

### `Krea2EditEmptyLatent` and `Krea2EditVAEDecode`

Create and decode target canvases at the nearest 16px grid. Requested target size is
independent of reference dimensions.

## Compatibility

This node pack and its paired trainer define one new architecture. Do not combine them
with LoRAs trained by the original upstream trainer, ai-toolkit's built-in Krea 2 edit
mode, or released target-fitted/cropped Identity Edit checkpoints: their reference
geometry and conditioning contract differ.
