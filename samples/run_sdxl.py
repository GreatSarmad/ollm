"""SDXL sample that reuses existing single-file weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ollm import Inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an SDXL prompt with low VRAM")
    parser.add_argument("prompt", help="Text prompt to render")
    parser.add_argument("--output", default="sdxl_sample.png", help="Where to save the image")
    parser.add_argument("--steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--models-dir", default="./models", help="Directory that already contains sd_xl_base_1.0.safetensors")
    parser.add_argument("--log-metrics", action="store_true", help="Print adapter metadata and CUDA peak memory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    inference = Inference(
        "sdxl-base-1.0",
        device=device,
        logging=True,
    )

    if args.log_metrics and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inference.ini_model(models_dir=args.models_dir)

    result = inference.generate(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        output_type="pil",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved image to {output_path}")

    if args.log_metrics and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA allocation: {peak_gb:.2f} GB")
    if args.log_metrics:
        print("Adapter metadata:", inference.adapter.metadata())


if __name__ == "__main__":
    main()
