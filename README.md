# oLLM Diffusion Toolkit (GreatSarmad fork)

This fork trims oLLM down to the pieces required for **running heavy diffusion checkpoints on small GPUs**.  The focus is on
streaming **SDXL**, **FLUX.1-dev**, and **Qwen Image Edit 2509** with 8–12 GB of VRAM by leaning on sequential CPU offload,
attention slicing, and VAE tiling.  No LLM samples or extra checkpoints are bundled here—the repository stays lightweight and
uses the weights you already have on disk.

## Key features

- Automatic detection of local SDXL single-file weights (`sd_xl_base_1.0.safetensors`) so they are loaded without another
  download.
- Native FLUX.1-dev and Qwen Image Edit adapters that apply the lowest-VRAM presets (sequential offload, tiling, chunking) on
  every run.
- Dedicated demo scripts in `samples/` for each supported checkpoint plus a generic CLI (`samples/run_diffusion.py`) with VRAM
  logging.

## Supported checkpoints

| Adapter ID        | Weight layout expected in `./models`                               | Notes |
| ----------------- | ------------------------------------------------------------------ | ----- |
| `sdxl-base-1.0`   | `sd_xl_base_1.0.safetensors` (single file)                         | Loads via `DiffusionPipeline.from_single_file` |
| `flux-1-dev`      | Diffusers directory (`model_index.json`, `unet/`, `vae/`, …)       | Works with locally extracted Hugging Face dump |
| `qwen-image-edit` | Diffusers directory (downloaded on demand from Hugging Face/CivitAI) | Requires Hugging Face access token for gated repo |

## 1. Install the toolkit

```bash
git clone https://github.com/GreatSarmad/ollm.git
cd ollm

# Base requirements (PyTorch 2.8 wheels are available on Colab and recent CUDA toolkits)
pip install -e .

# Optional low-VRAM extras (xFormers, SciPy, etc.)
pip install -e .[diffusion]
```

> **Tip:** Sequential CPU offload relies on `accelerate`.  You do *not* need `flash-attn` or other LLM-only packages—the
> dependency list is trimmed to diffusion use cases.

## 2. Point the repo at your existing weights

```
ollm/
├── models/
│   ├── sd_xl_base_1.0.safetensors      # already downloaded single-file SDXL
│   └── flux-1-dev/                     # extracted diffusers directory from Hugging Face
│       ├── model_index.json
│       ├── text_encoder/
│       ├── tokenizer_2/
│       └── ...
```

- **SDXL**: drop the 6.9 GB `sd_xl_base_1.0.safetensors` into `./models/`.  The adapter detects it automatically—no re-download
  occurs unless you pass `force_download=True`.
- **FLUX**: extract the `black-forest-labs/FLUX.1-dev` snapshot into `./models/flux-1-dev/` (matching the structure above).
- **Qwen Image Edit**: weights are not bundled.  On first run the adapter downloads from Hugging Face (requires `huggingface_hub`
  login) or from a CivitAI direct link provided via `download_url=` or the `OLLMDIFF_QWEN_IMAGE_EDIT_URL` environment variable.

## 3. Run the focused demos

All demos print adapter metadata and peak CUDA usage so you can confirm sequential offload is active.

```bash
# SDXL text-to-image using local single-file weights
python samples/run_sdxl.py "a serene mountain landscape at sunset" --log-metrics

# FLUX.1-dev high-resolution render
python samples/run_flux.py --log-metrics

# Qwen Image Edit (requires your own source image)
python samples/run_qwen_image_edit.py ./my_image.png "Replace the sky with a golden sunset" --log-metrics
```

By default the scripts look for weights under `./models`.  Use `--models-dir` to point elsewhere.

## 4. General-purpose CLI

For custom pipelines or batch jobs call the generic driver:

```bash
python samples/run_diffusion.py qwen-image-edit "Refine the sky with a golden sunset" \
    --image input.png --output sunset.png --num-steps 18 --guidance 4.5 \
    --forward-chunk 2 --attention-slicing auto --log-metrics
```

The important switches map directly to `DiffusionOptimizationConfig`:

- `--no-sequential-offload` / `--no-vae-tiling` / `--no-attention-slicing` toggle the low-VRAM presets.
- `--text-encoder-on-gpu` keeps prompt encoders resident if you have VRAM to spare.
- `--log-metrics` prints adapter metadata and the CUDA peak allocation right after the render.

## 5. Hugging Face & CivitAI tips

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

- Qwen Image Edit is a gated model; authenticate once per machine before calling the demos.
- To use a private CivitAI mirror, set `export OLLMDIFF_QWEN_IMAGE_EDIT_URL="https://civitai.com/api/download/..."` before
  running the scripts.  Archives are extracted into `./models/qwen-image-edit/` automatically.

## Troubleshooting checklist

1. **Model path mismatch** – call `Inference(...).adapter.metadata()` to verify which file/directory was used.
2. **High VRAM usage** – ensure `--log-metrics` reports `sequential_cpu_offload=True`.  If not, double-check that `accelerate`
   is installed and that you did not disable offload via flags.
3. **Missing CUDA** – every script falls back to CPU, but renders will be slow.  Install the correct PyTorch build for your GPU.

That’s it—you now have a lean toolkit for running SDXL, FLUX.1-dev, and Qwen Image Edit on hardware that normally tops out at 8–12 GB.
