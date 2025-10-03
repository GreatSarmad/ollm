from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, replace
from typing import Dict, Optional, TYPE_CHECKING

import torch

from ..base import PipelineAdapter
from ..registry import register_adapter
from .optimizations import (
    DiffusionOptimizationConfig,
    apply_diffusion_optimizations,
    build_optimizations,
)
from .runner import DiffusionRunner

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from diffusers import DiffusionPipeline


@dataclass
class DiffusionModelConfig:
    model_ids: tuple
    repo_id: Optional[str] = None
    download_url: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: torch.dtype = torch.float16
    enable_cpu_offload: bool = True
    enable_sequential_offload: bool = True
    enable_vae_tiling: bool = True
    enable_attention_slicing: bool = True
    attention_slicing: Optional[object] = "auto"
    forward_chunk_size: Optional[int] = 2
    enable_xformers: bool = False
    text_encoder_offload: str = "cpu"
    scheduler_override: Optional[str] = None
    single_file: Optional[str] = None # Added for single-file support


_DIFFUSION_MODELS: Dict[str, DiffusionModelConfig] = {
    "sdxl-base-1.0": DiffusionModelConfig(
        model_ids=("sdxl-base-1.0",),
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        single_file="sd_xl_base_1.0.safetensors", # Added single-file path
    ),
    "qwen-image-edit": DiffusionModelConfig(
        model_ids=("qwen-image-edit", "Qwen/Qwen-Image-Edit-2509"),
        repo_id="Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.float16,
    ),
    "flux-1-dev": DiffusionModelConfig(
        model_ids=("flux-1-dev", "FLUX.1-dev", "black-forest-labs/FLUX.1-dev"),
        repo_id="black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        attention_slicing="max",
        forward_chunk_size=2,
    ),
}


def _maybe_download_zip(url: str, destination_dir: str) -> None:
    if not url:
        raise ValueError(
            "No download URL configured. Set OLLMDIFF_QWEN_IMAGE_EDIT_URL to a direct download for the diffusers weights."
        )
    os.makedirs(destination_dir, exist_ok=True)
    filename = url.split("/")[-1] or "weights.zip"
    zip_path = os.path.join(destination_dir, filename)
    print(f"Downloading diffusion weights from {url} ...")

    import requests

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {zip_path}")
    print("Unpacking archive ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    os.remove(zip_path)


@register_adapter([model_id for config in _DIFFUSION_MODELS.values() for model_id in config.model_ids])
class DiffusionPipelineAdapter(PipelineAdapter):
    """Adapter that loads diffusion pipelines with aggressive offloading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_model_id = self.model_id
        self.config = self._resolve_config()
        self.model_source_path: Optional[str] = None # Added for metadata tracking
        self.model_source_kind = "unknown" # Added for metadata tracking
        if self.requested_model_id != self.cache_id:
            print(
                f"Redirecting requested diffusion id '{self.requested_model_id}' to registered '{self.cache_id}'"
            )
        overrides = {}
        for field in (
            "download_url",
            "repo_id",
            "revision",
            "variant",
            "torch_dtype",
            "enable_cpu_offload",
            "enable_sequential_offload",
            "enable_vae_tiling",
            "enable_attention_slicing",
            "attention_slicing",
            "forward_chunk_size",
            "enable_xformers",
            "text_encoder_offload",
            "scheduler_override",
        ):
            if field in self.kwargs and self.kwargs[field] is not None:
                overrides[field] = self.kwargs[field]
        if overrides:
            self.config = replace(self.config, **overrides)
        opt_defaults = {
            "sequential_cpu_offload": self.config.enable_sequential_offload,
            "model_cpu_offload": self.config.enable_cpu_offload,
            "enable_vae_tiling": self.config.enable_vae_tiling,
            "enable_attention_slicing": self.config.enable_attention_slicing,
            "attention_slicing": self.config.attention_slicing,
            "forward_chunk_size": self.config.forward_chunk_size,
            "enable_xformers": self.config.enable_xformers,
            "text_encoder_offload": self.config.text_encoder_offload,
        }
        opt_overrides = {
            key: self.kwargs[key]
            for key in DiffusionOptimizationConfig.__annotations__.keys()
            if key in self.kwargs
        }
        opt_defaults.update(opt_overrides)
        self.optimizations = build_optimizations(**opt_defaults)
        self.runner: Optional[DiffusionRunner] = None

    def _resolve_config(self) -> DiffusionModelConfig:
        if self.model_id in _DIFFUSION_MODELS:
            self.cache_id = self.model_id
            return _DIFFUSION_MODELS[self.model_id]

        for cache_id, config in _DIFFUSION_MODELS.items():
            if self.model_id in config.model_ids:
                self.cache_id = cache_id
                return config

        raise KeyError(f"Unknown diffusion model '{self.model_id}'")

    def prepare(self, models_dir: str, force_download: bool = False) -> str:
        os.makedirs(models_dir, exist_ok=True)
        model_dir = os.path.join(models_dir, self.cache_id)

        # START single-file logic
        if self.config.single_file and not force_download:
            single_path = os.path.join(models_dir, self.config.single_file)
            if os.path.isfile(single_path):
                print(f"Using existing single-file weights at {single_path}")
                self.model_source_path = single_path
                self.model_source_kind = "single-file"
                return single_path
        # END single-file logic

        if os.path.exists(model_dir) and not force_download:
            print(f"Using cached diffusion weights in {model_dir}")
            self.model_source_path = model_dir # Added for metadata tracking
            self.model_source_kind = "directory" # Added for metadata tracking
            return model_dir

        if os.path.exists(model_dir) and force_download:
            print(f"Removing existing model directory {model_dir} for fresh download")
            import shutil

            shutil.rmtree(model_dir)

        # Set model source path/kind before download for directory-based models
        self.model_source_path = model_dir
        self.model_source_kind = "directory"

        sanitized = self.cache_id.upper().replace("-", "_").replace("/", "_")

        if self.config.repo_id:
            from huggingface_hub import snapshot_download

            print(f"Downloading {self.config.repo_id} ...")
            snapshot_download(
                repo_id=self.config.repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                revision=self.config.revision,
            )
        else:
            download_url = self.config.download_url
            if download_url is None:
                env_key = f"OLLMDIFF_{sanitized}_URL"
                download_url = os.environ.get(env_key)
            if download_url:
                _maybe_download_zip(download_url, model_dir)
            else:
                raise ValueError(
                    f"Model '{self.requested_model_id}' has no repository or download URL configured. "
                    "Provide download_url=... or set the environment variable "
                    f"OLLMDIFF_{sanitized}_URL"
                )

        return model_dir

    def load(self, model_dir: str) -> None:
        from diffusers import DiffusionPipeline

        print(f"Loading diffusion pipeline from {model_dir}")
        # START loading logic for single-file vs directory
        if os.path.isdir(model_dir):
            pipeline = DiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=self.config.torch_dtype,
                use_safetensors=True,
                variant=self.config.variant,
            )
        else:
            pipeline = DiffusionPipeline.from_single_file(
                model_dir,
                torch_dtype=self.config.torch_dtype,
                use_safetensors=True,
            )
        # END loading logic
        
        apply_diffusion_optimizations(pipeline, self.optimizations, self.device)

        self.model = pipeline
        self.runner = DiffusionRunner(
            pipeline=pipeline,
            device=self.device,
            torch_dtype=self.config.torch_dtype,
            scheduler_override=self.config.scheduler_override,
        )

    def generate(self, *args, **kwargs):
        if self.runner is None:
            raise RuntimeError("Pipeline is not loaded")
        return self.runner.generate(*args, **kwargs)

    def metadata(self) -> Dict[str, str]:
        meta: Dict[str, str] = {"type": "diffusion", "model_id": self.cache_id}
        # Added model path/source tracking
        if self.model_source_path:
            meta["model_path"] = self.model_source_path
        meta["model_source"] = self.model_source_kind
        
        meta.update({
            "sequential_cpu_offload": str(self.optimizations.sequential_cpu_offload),
            "attention_slicing": str(self.optimizations.attention_slicing),
            "vae_tiling": str(self.optimizations.enable_vae_tiling),
            "forward_chunk_size": str(self.optimizations.forward_chunk_size),
            "text_encoder_offload": str(self.optimizations.text_encoder_offload),
        })
        return meta