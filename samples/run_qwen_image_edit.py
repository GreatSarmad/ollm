"""Qwen Image Edit demo that expects an input image on disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ollm import Inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Qwen Image Edit on a local file")
    parser.add_argument("image", help="Path to the source image (RGB)")
    parser.add_argument("prompt", help="Edit instruction to apply to the image")
    parser.add_argument("--output", default="qwen_edit.png", help="Destination path for the edited result")
    parser.add_argument("--strength", type=float, default=0.55, help="Denoising strength (0 = keep original, 1 = fully regenerate)")
    parser.add_argument("--steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--models-dir", default="./models", help="Directory used to cache Qwen Image Edit weights")
    parser.add_argument("--log-metrics", action="store_true", help="Print adapter metadata and CUDA peak memory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    source = Path(args.image)
    if not source.exists():
        raise FileNotFoundError(f"Input image not found: {source}")

    inference = Inference(
        "qwen-image-edit",
        device=device,
        logging=True,
    )

    if args.log_metrics and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inference.ini_model(models_dir=args.models_dir)

    init_image = Image.open(source).convert("RGB")

    result = inference.generate(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_type="pil",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved edited image to {output_path}")

    if args.log_metrics and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA allocation: {peak_gb:.2f} GB")
    if args.log_metrics:
        print("Adapter metadata:", inference.adapter.metadata())


if __name__ == "__main__":
    main()
