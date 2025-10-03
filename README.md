Of course! Here is the content with the conflict markers removed, keeping the parts that appear to be the final versions and integrating the remaining options logically.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png">
    <img alt="vLLM" src="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png" width=52%>
  </picture>
</p>

<h3 align="center">
LLM Inference for Large-Context Offline Workloads
</h3>

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like gpt-oss-20B, qwen3-next-80B or Llama-3.1-8B-Instruct on 100k context using ~$200 consumer GPU with 8GB VRAM. No quantization is used—only fp16/bf16 precision.
With 0.6.0, the same streaming/offload machinery now works for diffusion pipelines: large image or video checkpoints can be executed on 8–12 GB cards via CPU and disk orchestration.

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

8GB Nvidia 3060 Ti Inference memory usage:
Model	Weights	Context length	KV cache	Baseline VRAM (no offload)	oLLM GPU VRAM	oLLM Disk (SSD)
qwen3-next-80B	160 GB (bf16)	50k	20 GB	~190 GB	~7.5 GB	180 GB
gpt-oss-20B	13 GB (packed bf16)	10k	1.4 GB	~40 GB	~7.3GB	15 GB
gemma3-12B	25 GB (bf16)	50k	18.5 GB	~45 GB	~6.7 GB	43 GB
llama3-1B-chat	2 GB (fp16)	100k	12.6 GB	~16 GB	~5 GB	15 GB
llama3-3B-chat	7 GB (fp16)	100k	34.1 GB	~42 GB	~5.3 GB	42 GB
llama3-8B-chat	16 GB (fp16)	100k	52.4 GB	~71 GB	~6.6 GB	69 GB

Export to Sheets
<small>By "Baseline" we mean typical inference without any offloading</small>

How do we achieve this:

Loading layer weights from SSD directly to GPU one by one

Offloading KV cache to SSD and loading back directly to GPU, no quantization or PagedAttention

Offloading layer weights to CPU if needed

FlashAttention-2 with online softmax. Full attention matrix is never materialized.

Chunked MLP. Intermediate upper projection layers may get large, so we chunk MLP as well

Typical use cases include:

Analyze contracts, regulations, and compliance reports in one pass

Summarize or extract insights from massive patient histories or medical literature

Process very large log files or threat reports locally

Analyze historical chats to extract the most common issues/questions users have

Supported Nvidia GPUs: Ampere (RTX 30xx, A30, A4000, A10), Ada Lovelace (RTX 40xx, L4), Hopper (H100), and newer

Getting Started
It is recommended to create venv or conda environment first

Bash

python3 -m venv ollm_env
source ollm_env/bin/activate
Install oLLM with pip install ollm or from source:

Bash

git clone https://github.com/Mega4alik/ollm.git
cd ollm
pip install -e .
pip install kvikio-cu{cuda_version} Ex, kvikio-cu12
💡 Note
qwen3-next requires 4.57.0.dev version of transformers to be installed as pip install git+https://github.com/huggingface/transformers.git

Example
Code snippet sample

Bash

from ollm import Inference, file_get_contents, TextStreamer
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
or run sample python script as PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example.py

More samples

gemma3-12B image+text 

Stable Diffusion XL / Qwen Image Edit on 8 GB GPUs

FLUX.1-dev streaming demo

Diffusion pipelines on small GPUs
Large diffusion checkpoints behave very differently from autoregressive LLMs: there is no KV cache to offload, but the UNet +
text encoder + VAE stack can easily exceed 60 GB in FP16. The diffusion adapter mirrors oLLM’s SSD-first philosophy by keeping
all large modules on CPU/disk and only staging the active block on GPU via diffusers’ sequential offload APIs. Combined with
attention slicing, VAE tiling, and optional xFormers attention, Qwen Image Edit and SDXL now run end-to-end on 8–12 GB cards.

What changes under the hood?
Optimisation	Purpose	Where it lives
Sequential CPU offload	Streams UNet/VAE blocks between CPU and GPU, mimicking oLLM’s layer streaming	DiffusionOptimizationConfig.sequential_cpu_offload
Attention slicing / windowing	Bounds memory of spatial attention for 1024×1024 renders	attention_slicing, max_attention_window
VAE tiling & slicing	Decodes large canvases in overlapping tiles to avoid activation spikes	enable_vae_tiling, enable_vae_slicing
Prompt embedding cache	Encodes prompts once and reuses tensors across batches/runs	DiffusionRunner
Optional xFormers / channels-last	Uses memory-efficient kernels when available	enable_xformers, enable_channels_last

Export to Sheets
These knobs are exposed via keyword arguments on Inference or from the sample CLI (see below). By default the adapter keeps
text encoders on CPU—only the UNet is migrated to GPU per denoising step—so VRAM usage stays within ~7–8 GB even for the
19 GB Qwen Image Edit checkpoint.

Running SDXL or Qwen Image Edit programmatically
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
Need a sanity check before tackling 20 B parameter checkpoints? samples/run_flux.py loads the official FLUX.1-dev weights via the same adapter and prints peak GPU usage plus the optimisation plan so you can confirm sequential offload is active.

Need weights from a private CivitAI mirror? Pass download_url=... to Inference or set the environment variable
OLLMDIFF_QWEN_IMAGE_EDIT_URL. ZIP archives are extracted into ./models/<model-id> automatically as long as they contain the
standard diffusers layout (model_index.json, unet, vae, ...).

CLI: inspect VRAM-friendly presets
python samples/run_diffusion.py qwen-image-edit "Refine the sky with a golden sunset" \
    --image input.png --output sunset.png --num-steps 18 --guidance 4.5 \
    --num-images 2 --clip-skip 2 --guidance-rescale 0.7 \
    --forward-chunk 2 --attention-slicing auto --log-metrics
Flags such as --no-sequential-offload, --xformers, --no-vae-tiling, --no-attention-slicing, and --text-encoder-on-gpu map directly to the
DiffusionOptimizationConfig dataclass. Pass --log-metrics to print the adapter metadata and peak CUDA allocation once the
render finishes—handy when validating Colab runs—or call Inference(...).adapter.metadata() directly in your own scripts.

Expected VRAM with defaults (RTX 3060 Ti, 8 GB)
Model	Precision	Steps	Peak VRAM	Notes
SDXL Base 1.0	fp16	30	~7.2 GB	Sequential offload + tiling
Qwen Image Edit 2509	fp16	20	~7.6 GB	Text encoder kept on CPU, UNet streamed
Qwen Image Edit 2509	fp16	8	~6.1 GB	Lightning-style LoRA (fast draft renders)

Export to Sheets
Reducing steps (e.g., Lightning LoRAs), enabling xFormers, or rendering at 768×768 further lowers memory pressure. For extreme
constraints you can disable classifier-free guidance or fall back to CPU generation—the adapter reuses the same code paths.

⚠️ CivitAI archives should include the diffusers folder structure. If you see model_index.json alongside unet/ and vae/, the
auto-extractor will place them correctly under ./models/<model-id>.

Roadmap
For visibility of what's coming next (subject to change)

Voxtral-small-24B ASR model coming on Oct 5, Sun

Qwen3-VL or alternative vision model by Oct 12, Sun

Qwen3-Next MultiTokenPrediction in R&D

Efficient weight loading in R&D

Contact us
If there’s a model you’d like to see supported, feel free to reach out at anuarsh@ailabs.us—I’ll do my best to make it happen.

The video below offers a hands-on demonstration of how to install and run the oLLM library to enable large-context LLM inference on consumer GPUs. oLLM - Run 80GB Model on 8GB VRAM Locally - Hands-on Demo