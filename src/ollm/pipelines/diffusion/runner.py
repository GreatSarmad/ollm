from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


@dataclass
class DiffusionRunConfig:
    prompt: Union[str, Sequence[str]]
    negative_prompt: Optional[Union[str, Sequence[str]]] = None
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    height: Optional[int] = None
    width: Optional[int] = None
    generator: Optional[Union[int, torch.Generator]] = None
    output_type: str = "pil"
    eta: Optional[float] = None
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None
    strength: Optional[float] = None


class DiffusionRunner:
    """Executes diffusion denoising loops with resource-aware defaults."""

    def __init__(
        self,
        pipeline,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float16,
        scheduler_override: Optional[str] = None,
    ) -> None:
        self.pipeline = pipeline
        self.device = device
        self.torch_dtype = torch_dtype
        self._apply_scheduler_override(scheduler_override)
        self.pipeline.set_progress_bar_config(leave=False)

    def _apply_scheduler_override(self, scheduler_name: Optional[str]) -> None:
        if not scheduler_name:
            return
        from importlib import import_module

        module = import_module("diffusers.schedulers")
        scheduler_cls = getattr(module, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler '{scheduler_name}'")
        self.pipeline.scheduler = scheduler_cls.from_config(self.pipeline.scheduler.config)

    def _prepare_generator(self, generator) -> Optional[torch.Generator]:
        if generator is None:
            return None
        if isinstance(generator, torch.Generator):
            return generator
        if isinstance(generator, int):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(generator)
            return gen
        raise TypeError("generator must be None, an int seed, or a torch.Generator instance")

    def generate(self, config: Optional[DiffusionRunConfig] = None, **kwargs):
        if config is None:
            config = DiffusionRunConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a DiffusionRunConfig instance or keyword arguments, not both")

        generator = self._prepare_generator(config.generator)

        return self.pipeline(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            generator=generator,
            output_type=config.output_type,
            eta=config.eta,
            denoising_start=config.denoising_start,
            denoising_end=config.denoising_end,
            strength=config.strength,
        )
