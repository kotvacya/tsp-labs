"""Реализация модели UniDepth (Universal Monocular Metric Depth Estimation, CVPR 2024 / 2025)."""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from .base import BaseModel


class UniDepthModel(BaseModel):
    """Обертка для модели UniDepth (UniDepthV2 / UniDepthV1).
    
    Поддерживает:
      - UniDepthV2 (ViT-L/14, ViT-B/14, ViT-S/14)
      - Прямую загрузку из локальных чекпоинтов или HuggingFace Hub
      - Работу на ROCm (AMD GPU) и CUDA
      - Предсказание метрической глубины для произвольных разрешений
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.variant = config.get('variant', 'v2-vitl14')
        self.version = config.get('version', 'v2' if 'v2' in self.variant.lower() else 'v1')
        self.min_depth = float(config.get('min_depth', 1e-3))
        default_max = 80.0 if 'kitti' in str(config.get('checkpoint', '')).lower() else 10.0
        self.max_depth = float(config.get('max_depth', default_max))
        self.resolution_level: Optional[int] = config.get('resolution_level', None)
        self.interpolation_mode = config.get('interpolation_mode', 'bilinear')

        # 1. Импорт библиотеки unidepth с fallback на внешний путь
        try:
            from unidepth.models import UniDepthV1, UniDepthV2
        except ImportError:
            code_path = config.get('unidepth_code_path', '/home/kot/dev/python/UniDepth')
            if code_path and os.path.exists(code_path):
                if str(code_path) not in sys.path:
                    sys.path.insert(0, str(code_path))
            try:
                from unidepth.models import UniDepthV1, UniDepthV2
            except ImportError as e:
                raise ImportError(
                    f"Не удалось импортировать unidepth ни из окружения, ни из пути '{code_path}'. "
                    f"Ошибка: {e}"
                )

        if self.version == 'v2':
            self._model_cls = UniDepthV2
        else:
            self._model_cls = UniDepthV1

        # 2. Определение пути к чекпоинту
        checkpoint_path = config.get('checkpoint')
        if not checkpoint_path:
            # Дефолтный локальный путь или hub ID
            local_default = Path("checkpoints/unidepth/unidepth-v2-vitl14")
            if local_default.exists():
                checkpoint_path = str(local_default)
            else:
                checkpoint_path = f"lpiccinelli/unidepth-{self.variant}"

        self.load_weights(checkpoint_path)

        # Конфигурация параметров инференса модели
        if hasattr(self.model, 'interpolation_mode'):
            self.model.interpolation_mode = self.interpolation_mode
        if self.resolution_level is not None and hasattr(self.model, 'resolution_level'):
            self.model.resolution_level = self.resolution_level

    def load_weights(self, checkpoint_path: str) -> None:
        """Загрузка модели из локальной директории или HuggingFace."""
        resolved = Path(checkpoint_path).expanduser().resolve()
        
        # Если передан путь к файлу (например model.safetensors), берем родительскую папку
        target_path: str = str(resolved) if resolved.is_dir() else (str(resolved.parent) if resolved.is_file() else checkpoint_path)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model = self._model_cls.from_pretrained(target_path)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Предобработка: приведение к [0, 255] для встроенного пайплайна model.infer()."""
        # В бенчмарке тензор в диапазоне [0, 1]
        if image.max() <= 1.0 + 1e-4:
            return image * 255.0
        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Прямой проход: RGB [B, 3, H, W] -> метрическая глубина [B, 1, H, W]."""
        with torch.no_grad():
            preds = self.model.infer(image)
        return preds['depth']

    def postprocess(self, depth: torch.Tensor) -> torch.Tensor:
        """Постобработка: ограничение по диапазону глубин (clipping)."""
        return torch.clamp(depth, min=self.min_depth, max=self.max_depth)

    @property
    def name(self) -> str:
        return f"UniDepth-{self.variant}"
