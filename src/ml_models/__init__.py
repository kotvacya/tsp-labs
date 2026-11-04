"""Регистр и фабрика моделей монокулярной оценки глубины."""

from typing import Any, Dict, Type
from .base import BaseModel
from .iebins import IEBinsModel
from .adabins import AdaBinsModel
from .depth_anything import DepthAnythingModel
from .unidepth import UniDepthModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'iebins': IEBinsModel,
    'adabins': AdaBinsModel,
    'depth_anything': DepthAnythingModel,
    'unidepth': UniDepthModel,
}


def get_model(config: Dict[str, Any]) -> BaseModel:
    """Создание экземпляра модели по конфигу."""
    name = config.get('name', '').lower()
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(f"Неизвестная модель: '{name}'. Доступны: {available}")
    return MODEL_REGISTRY[name](config)


__all__ = ['BaseModel', 'IEBinsModel', 'AdaBinsModel', 'DepthAnythingModel', 'UniDepthModel', 'MODEL_REGISTRY', 'get_model']
