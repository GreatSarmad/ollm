<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png">
    <img alt="vLLM" src="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png" width=52%>
  </picture>
</p>

<h3 align="center">
LLM Inference for Large-Context Offline Workloads
</h3>

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like gpt-oss-20B, qwen3-next-80B, or Llama-3.1-8B-Instruct on 100k context using a consumer GPU with 8GB VRAM. No quantization is used—only fp16/bf16 precision.

With 0.6.0, the same streaming/offload machinery now works for diffusion pipelines: large image or video checkpoints can be executed on 8–12 GB cards via CPU and disk orchestration.

The GreatSarmad fork of oLLM trims the toolkit down to the pieces required for running heavy diffusion checkpoints on small GPUs. The focus is on streaming SDXL, FLUX.1-dev, and Qwen Image Edit 2509 with 8–12 GB of VRAM by leaning on sequential CPU offload, attention slicing, and VAE tiling. No LLM samples or extra checkpoints are bundled here—the repository stays lightweight and uses the weights you already have on disk.

<p dir="auto"><em>Latest updates (0.6.0)</em> 🔥</p>
<ul dir="auto">
<li>Pluggable pipeline runtime with first-class diffusion support (run Stable Diffusion XL, Qwen Image Edit, or FLUX on 8–12 GB GPUs)</li>
<li>Multimodal gemma3-12B (image+text) added. <a href="https://github.com/Mega4alik/ollm/blob/main/example_multimodality.py">[sample with image]</a> </li>
<li>.safetensor files are now read without mmap so they no longer consume RAM through page cache</li>
<li>qwen3-next-80B DiskCache support added</li>
<li>qwen3-next-80B (160GB model) added with <span style="color:blue">⚡️1tok/2s</span> throughput (our fastest model so far)</li>
<li>gpt-oss-20B flash-attention-like implementation added to reduce VRAM usage </li>
<li>gpt-oss-20B chunked MLP added to reduce VRAM usage </li>
</ul>

Performance and Use Cases
LLM Inference Memory Usage (8GB Nvidia 3060 Ti)
Model	Weights	Context length	KV cache	Baseline VRAM (no offload)	oLLM GPU VRAM	oLLM Disk (SSD)
qwen3-next-80B	160 GB (bf16)	50k	20 GB	~190 GB	~7.5 GB	180 GB
gpt-oss-20B	13 GB (packed bf16)	10k	1.4 GB	~40 GB	~7.3GB	15 GB
gemma3-12B	25 GB (bf16)	50k	18.5 GB	~45 GB	~6.7 GB	43 GB
llama3-1B-chat	2 GB (fp16)	100k	12.6 GB	~16 GB	~5 GB	15 GB
llama3-3B-chat	7 GB (fp16)	100k	34.1 GB	~42 GB	~5.3 GB	42 GB
llama3-8B-chat	16 GB (fp16)	100k	52.4 GB	~71 GB	~6.6 GB	69 GB

Export to Sheets
<small>By "Baseline" we mean typical inference without any offloading.</small>

Diffusion Expected VRAM with Defaults (RTX 3060 Ti, 8 GB)
Model	Precision	Steps	Peak VRAM	Notes
SDXL Base 1.0	fp16	30	~7.2 GB	Sequential offload + tiling
Qwen Image Edit 2509	fp16	20	~7.6 GB	Text encoder kept on CPU, UNet streamed
Qwen Image Edit 2509	fp16	8	~6.1 GB	Lightning-style LoRA (fast draft renders)

Export to Sheets
How oLLM Achieves This
Loading layer weights from SSD directly to GPU one by one.

Offloading KV cache to SSD and loading back directly to GPU, without quantization or PagedAttention.

Offloading layer weights to CPU if needed.

FlashAttention-2 with online softmax. Full attention matrix is never materialized.

Chunked MLP. Intermediate upper projection layers are chunked to reduce VRAM usage.

Typical Use Cases
Analyze contracts, regulations, and compliance reports in one pass.

Summarize or extract insights from massive patient histories or medical literature.

Process very large log files or threat reports locally.

Analyze historical chats to extract the most common issues/questions users have.

Run heavy diffusion checkpoints (SDXL, FLUX.1-dev, Qwen Image Edit) on small GPUs.

Supported Checkpoints (Diffusion)
Adapter ID	Weight layout expected in ./models	Notes
sdxl-base-1.0	sd_xl_base_1.0.safetensors (single file)	Loads via DiffusionPipeline.from_single_file. Automatic detection of local weights.
flux-1-dev	Diffusers directory (model_index.json, unet/, vae/, …)	Works with locally extracted Hugging Face dump.
qwen-image-edit	Diffusers directory (downloaded on demand from Hugging Face/CivitAI)	Requires Hugging Face access token for gated repo.

Export to Sheets
Diffusion Optimization Configuration
Large diffusion checkpoints behave very differently from autoregressive LLMs. The diffusion adapter mirrors oLLM’s SSD-first philosophy by keeping all large modules on CPU/disk and only staging the active block on GPU via diffusers’ sequential offload APIs.

Optimisation	Purpose	Where it lives
Sequential CPU offload	Streams UNet/VAE blocks between CPU and GPU, mimicking oLLM’s layer streaming.	DiffusionOptimizationConfig.sequential_cpu_offload
Attention slicing / windowing	Bounds memory of spatial attention for 1024×1024 renders.	attention_slicing, max_attention_window
VAE tiling & slicing	Decodes large canvases in overlapping tiles to avoid activation spikes.	enable_vae_tiling, enable_vae_slicing
Prompt embedding cache	Encodes prompts once and reuses tensors across batches/runs.	DiffusionRunner
Optional xFormers / channels-last	Uses memory-efficient kernels when available.	enable_xformers, enable_channels_last

Export to Sheets
Getting Started
It is recommended to create a venv or conda environment first. Supported Nvidia GPUs include: Ampere (RTX 30xx, A30, A4000, A10), Ada Lovelace (RTX 40xx, L4), Hopper (H100), and newer.

Bash

python3 -m venv ollm_env
source ollm_env/bin/activate
1. Install the Toolkit
Install oLLM with pip install ollm or from source:

Bash

git clone https://github.com/Mega4alik/ollm.git
cd ollm

# Base requirements (PyTorch 2.8 wheels are available on Colab and recent CUDA toolkits)
pip install -e .

# Optional low-VRAM extras (xFormers, SciPy, etc.) for diffusion
pip install -e .[diffusion]
Tip: Sequential CPU offload relies on accelerate. You do not need flash-attn or other LLM-only packages for diffusion use cases—the dependency list is trimmed.

Additional Requirements
qwen3-next requires the 4.57.0.dev version of transformers to be installed:

Bash

pip install git+https://github.com/huggingface/transformers.git
You may also need kvikio:

Bash

pip install kvikio-cu{cuda_version} # Ex, kvikio-cu12
2. Point the Repo at Your Existing Weights (Diffusion)
Organize your weights under the ./models/ directory:

ollm/
├── models/
│   ├── sd_xl_base_1.0.safetensors      # already downloaded single-file SDXL
│   └── flux-1-dev/                     # extracted diffusers directory from Hugging Face
│       ├── model_index.json
│       ├── text_encoder/
│       ├── tokenizer_2/
│       └── ...
SDXL: Drop the 6.9 GB sd_xl_base_1.0.safetensors into ./models/.

FLUX: Extract the black-forest-labs/FLUX.1-dev snapshot into ./models/flux-1-dev/.

Qwen Image Edit: Weights are not bundled. On first run the adapter downloads from Hugging Face (requires huggingface_hub login) or from a CivitAI direct link provided via download_url= or the OLLMDIFF_QWEN_IMAGE_EDIT_URL environment variable.

3. Example Usage (LLM)
Bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example.py
Code snippet sample:

Python

from ollm import Inference, TextStreamer
o = Inference("llama3-1B-chat", device="cuda:0", logging=True) #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #(optional) offload some layers to CPU for speed boost
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

messages = [{"role":"system", "content":"You are helpful AI assistant"}, {"role":"user", "content":"List planets"}]
input_ids = o.tokenizer.apply_chat_template(messages, reasoning_effort="minimal", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
outputs = o.model.generate(input_ids=input_ids, past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
More samples are available for gemma3-12B image+text, Stable Diffusion XL / Qwen Image Edit on 8 GB GPUs, and FLUX.1-dev streaming demo.

4. Example Usage (Diffusion)
Focused Demos
All demos print adapter metadata and peak CUDA usage so you can confirm sequential offload is active.

Bash

# SDXL text-to-image using local single-file weights
python samples/run_sdxl.py "a serene mountain landscape at sunset" --log-metrics

# FLUX.1-dev high-resolution render
python samples/run_flux.py --log-metrics

# Qwen Image Edit (requires your own source image)
python samples/run_qwen_image_edit.py ./my_image.png "Replace the sky with a golden sunset" --log-metrics
By default the scripts look for weights under ./models. Use --models-dir to point elsewhere.

General-purpose CLI
Bash

python samples/run_diffusion.py qwen-image-edit "Refine the sky with a golden sunset" \
    --image input.png --output sunset.png --num-steps 18 --guidance 4.5 \
    --forward-chunk 2 --attention-slicing auto --log-metrics
The important switches map directly to DiffusionOptimizationConfig:

--no-sequential-offload / --no-vae-tiling / --no-attention-slicing toggle the low-VRAM presets.

--text-encoder-on-gpu keeps prompt encoders resident if you have VRAM to spare.

--log-metrics prints adapter metadata and the CUDA peak allocation right after the render.

Programmatic Use
Python

from PIL import Image
from ollm import Inference

pipe = Inference(
    "qwen-image-edit",
    device="cuda:0",
    sequential_cpu_offload=True,   # default: stream UNet blocks from CPU
    attention_slicing="auto",      # cap attention memory
    forward_chunk_size=2,           # chunk UNet feed-forward ops
)
pipe.ini_model(models_dir="./models")

init_image = Image.open("input.png").convert("RGB")
result = pipe.generate(
    prompt="A watercolor skyline at dusk",
    image=init_image,
    strength=0.55,
    num_inference_steps=20,
    guidance_scale=4.0,
    num_images_per_prompt=2,
)

for idx, img in enumerate(result.images):
    img.save(f"edited_{idx}.png")
5. Hugging Face & CivitAI Tips
Qwen Image Edit is a gated model; authenticate once per machine before calling the demos.

Python

from huggingface_hub import login
login(token="hf_your_token_here")
To use a private CivitAI mirror, set export OLLMDIFF_QWEN_IMAGE_EDIT_URL="https://civitai.com/api/download/..." before running the scripts or pass download_url= to Inference. Archives are extracted into ./models/<model-id>/ automatically.

Troubleshooting Checklist
Model path mismatch – call Inference(...).adapter.metadata() to verify which file/directory was used.

High VRAM usage – ensure --log-metrics reports sequential_cpu_offload=True. If not, double-check that accelerate is installed and that you did not disable offload via flags.

Missing CUDA – every script falls back to CPU, but renders will be slow. Install the correct PyTorch build for your GPU.

CivitAI archives should include the diffusers folder structure (i.e., model_index.json alongside unet/ and vae/).

Roadmap
For visibility of what's coming next (subject to change):

Voxtral-small-24B ASR model coming on Oct 5, Sun

Qwen3-VL or alternative vision model by Oct 12, Sun

Qwen3-Next MultiTokenPrediction in R&D

Efficient weight loading in R&D

Contact us at anuarsh@ailabs.us if there’s a model you’d like to see supported.

The video below offers a hands-on demonstration of how to install and run the oLLM library to enable large-context LLM inference on consumer GPUs. oLLM - Run 80GB Model on 8GB VRAM Locally - Hands-on Demo