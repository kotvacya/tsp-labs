"""Заглушка для модели AdaBins (Bhat et al., CVPR 2021)."""

from typing import Any, Dict
import torch

from .base import BaseModel


class AdaBinsModel(BaseModel):
    """AdaBins (Adaptive Bins) — опциональная модель для бенчмарка."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.variant = config.get('variant', 'default')

    def load_weights(self, checkpoint_path: str) -> None:
        raise NotImplementedError("AdaBinsModel.load_weights будет реализован на этапе масштабирования.")

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AdaBinsModel.preprocess будет реализован на этапе масштабирования.")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AdaBinsModel.forward будет реализован на этапе масштабирования.")

    @property
    def name(self) -> str:
        return f"AdaBins-{self.variant}"
