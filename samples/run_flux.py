"""Run FLUX.1-dev with the oLLM diffusion adapter on a small GPU.

This sample mirrors the diagnostic workflow requested in Colab notebooks: it
loads the official ``black-forest-labs/FLUX.1-dev`` checkpoint through the
``Inference`` API, applies the sequential offload/tiling defaults, and prints
peak VRAM usage plus adapter metadata so it is easy to verify that the
oLLM-specific optimisations were active.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ollm import Inference


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inference = Inference(
        "flux-1-dev",
        device=device,
        logging=True,
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    inference.ini_model(models_dir="./models")

    result = inference.generate(
        prompt="a serene mountain landscape at sunset",
        num_inference_steps=20,
        guidance_scale=5.5,
        height=1024,
        width=1024,
        output_type="pil",
    )

    output_path = Path("flux_sample.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved image to {output_path}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA allocation: {peak_gb:.2f} GB")
    print("Adapter metadata:", inference.adapter.metadata())


if __name__ == "__main__":
    main()
