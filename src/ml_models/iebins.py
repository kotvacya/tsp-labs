"""Реализация модели IEBins (Iterative Elastic Bins, CVPR 2024)."""

import os
import sys
from pathlib import Path
from typing import Any, Dict
import torch

from .base import BaseModel


class IEBinsModel(BaseModel):
    """Обертка для модели IEBins (Swin-Large / Swin-Tiny)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.variant = config.get('variant', 'large07')
        if not self.variant.endswith('07'):
            # Нормализация 'large' -> 'large07', 'tiny' -> 'tiny07'
            self.variant = f"{self.variant}07"

        self.min_depth = float(config.get('min_depth', 1e-3))
        default_max = 80.0 if 'kitti' in str(config.get('checkpoint', '')).lower() else 10.0
        self.max_depth = float(config.get('max_depth', default_max))
        self.post_process_enabled = bool(config.get('post_process', False))
        self.do_kb_crop = bool(config.get('do_kb_crop', 'kitti' in str(config.get('checkpoint', '')).lower()))

        # 1. Приоритетный импорт из встроенного модуля backbones
        try:
            from .backbones.iebins.networks.NewCRFDepth import NewCRFDepth
            from .backbones.iebins import utils as iebins_utils
            self._network_cls = NewCRFDepth
            self._utils = iebins_utils if self.post_process_enabled else None
        except ImportError:
            # 2. Fallback на внешний путь (если задан)
            iebins_code_path = config.get('iebins_code_path')
            if iebins_code_path and os.path.exists(iebins_code_path):
                if str(iebins_code_path) not in sys.path:
                    sys.path.insert(0, str(iebins_code_path))
            try:
                from networks.NewCRFDepth import NewCRFDepth
                self._network_cls = NewCRFDepth
                if self.post_process_enabled:
                    try:
                        import utils as iebins_utils
                        self._utils = iebins_utils
                    except ImportError:
                        self._utils = None
            except ImportError as e:
                raise ImportError(
                    f"Не удалось загрузить NewCRFDepth ни из models.backbones.iebins, "
                    f"ни из внешнего пути '{iebins_code_path}'. Ошибка: {e}"
                )

        # Инициализация архитектуры
        self.model = self._network_cls(
            version=self.variant,
            inv_depth=False,
            max_depth=self.max_depth,
            pretrained=None
        )

        checkpoint_path = config.get('checkpoint')
        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str) -> None:
        """Загрузка весов из .pth файла."""
        resolved_path = Path(checkpoint_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {resolved_path}")

        # PyTorch 2.6+ требует weights_only=False для словарей чекпоинтов
        checkpoint = torch.load(resolved_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Удаление префикса 'module.' если обучение шло в DataParallel/DDP
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            cleaned_state_dict[key] = v

        self.model.load_state_dict(cleaned_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Нормализация входного RGB изображения [0, 1] по ImageNet mean/std."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Inference IEBins: RGB -> глубина в метрах [B, 1, H, W]."""
        outputs = self.model(image)
        pred_depths_r_list = outputs[0]
        pred_depth = pred_depths_r_list[-1]

        if self.post_process_enabled and self._utils is not None:
            # Test-time augmentation (горизонтальный flip)
            image_flipped = self._utils.flip_lr(image)
            outputs_flipped = self.model(image_flipped)
            pred_flipped = outputs_flipped[0][-1]
            pred_depth = self._utils.post_process_depth(pred_depth, pred_flipped)

        return pred_depth

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Инференс модели с автоматической адаптацией разрешения (KB crop для KITTI)."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        H, W = image.shape[-2:]
        if self.do_kb_crop and (H != 352 or W != 1216):
            # Стандартная обрезка (bottom-center crop) под входной размер сети 352x1216
            top_margin = int(H - 352)
            left_margin = int((W - 1216) / 2)
            image_cropped = image[..., top_margin:top_margin + 352, left_margin:left_margin + 1216]
            image_prep = self.preprocess(image_cropped).to(self.device)

            with torch.no_grad():
                pred_cropped = self.forward(image_prep)
            pred_cropped = self.postprocess(pred_cropped)

            # Восстановление исходного разрешения (uncropping)
            pred_full = torch.zeros(
                (image.shape[0], 1, H, W),
                device=pred_cropped.device,
                dtype=pred_cropped.dtype
            )
            pred_full[..., top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_cropped
            return pred_full

        return super().predict(image)

    @property
    def name(self) -> str:
        suffix = "-pp" if self.post_process_enabled else ""
        return f"IEBins-{self.variant}{suffix}"
