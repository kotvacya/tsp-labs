"""Заглушка для модели DepthAnything-v2 (Yang et al., 2024)."""

from typing import Any, Dict
import torch

from .base import BaseModel


class DepthAnythingModel(BaseModel):
    """DepthAnything-v2 — опциональная foundation model для бенчмарка."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.encoder = config.get('encoder', 'vitl')

    def load_weights(self, checkpoint_path: str) -> None:
        raise NotImplementedError("DepthAnythingModel.load_weights будет реализован на этапе масштабирования.")

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DepthAnythingModel.preprocess будет реализован на этапе масштабирования.")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DepthAnythingModel.forward будет реализован на этапе масштабирования.")

    @property
    def name(self) -> str:
        return f"DepthAnythingV2-{self.encoder}"
