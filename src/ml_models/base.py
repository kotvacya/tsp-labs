"""Абстрактный базовый класс для всех моделей монокулярной оценки глубины (MDE)."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class BaseModel(ABC):
    """Абстрактный класс MDE-модели."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Any = None

    @abstractmethod
    def load_weights(self, checkpoint_path: str) -> None:
        """Загрузка весов модели из чекпоинта."""
        pass

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Предобработка входного изображения (тензор shape [B, C, H, W] или [C, H, W])."""
        pass

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели: RGB -> карта глубины."""
        pass

    def postprocess(self, depth: torch.Tensor) -> torch.Tensor:
        """Постобработка карты глубины (по умолчанию возвращает без изменений)."""
        return depth

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Полный пайплайн предсказания: preprocess -> forward -> postprocess."""
        # Убеждаемся в наличии batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = self.preprocess(image).to(self.device)
        with torch.no_grad():
            depth = self.forward(image)
        depth = self.postprocess(depth)
        return depth

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя модели (для отчётов, логов и графиков)."""
        pass
