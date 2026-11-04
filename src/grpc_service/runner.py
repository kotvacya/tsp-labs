"""Вычислительный движок инференса нейросетевых моделей глубины."""

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from src.ml_models.metrics.depth_metrics import compute_depth_metrics

# Автоматическая настройка аппаратного ускорения ROCm для AMD GPU
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1030")


def detect_device() -> str:
    """Определение доступного вычислительного ускорителя."""
    if torch.cuda.is_available():
        return f"rocm/cuda:{torch.cuda.get_device_name(0)}"
    return "cpu (torch)"


def load_depth_map(file_path: Path) -> Optional[np.ndarray]:
    """Чтение ground truth карты глубины из PNG (16-бит)."""
    if not file_path.exists():
        return None
    img = Image.open(file_path)
    depth_raw = np.array(img, dtype=np.float32)

    # KITTI: глубина в метрах = raw / 256.0
    if "kitti" in str(file_path).lower():
        depth = depth_raw / 256.0
    # NYU Depth v2: глубина в метрах = raw / 1000.0 (миллиметры в метры)
    else:
        depth = depth_raw / 1000.0
    return depth


class ModelBenchmarkRunner:
    """Исполнитель инференса реальных нейросетевых моделей глубины."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path.cwd()
        self._loaded_models: Dict[Tuple[str, str, str], Any] = {}

    def _get_or_load_model(self, architecture: str, backbone: str, weights_rel_path: str) -> Any:
        key = (architecture.lower(), backbone.lower(), weights_rel_path)
        if key in self._loaded_models:
            return self._loaded_models[key]

        weights_path = self.root_dir / weights_rel_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

        arch = architecture.lower()
        is_large = "large" in backbone.lower() or "vit" in backbone.lower()
        is_kitti = "kitti" in weights_rel_path.lower()
        max_depth = 80.0 if is_kitti else 10.0

        from src.ml_models import get_model
        if "iebins" in arch or "iebins" in weights_rel_path.lower():
            variant = "large07" if is_large else "tiny07"
            cfg = {
                "name": "iebins",
                "variant": variant,
                "checkpoint": str(weights_path),
                "min_depth": 0.001,
                "max_depth": max_depth,
            }
            model = get_model(cfg)
            self._loaded_models[key] = model
            return model
        elif "unidepth" in arch or "unidepth" in weights_rel_path.lower():
            cfg = {
                "name": "unidepth",
                "variant": "v2-vitl14",
                "checkpoint": str(weights_path),
                "min_depth": 0.001,
                "max_depth": max_depth,
            }
            model = get_model(cfg)
            self._loaded_models[key] = model
            return model

        raise ValueError(f"Неизвестная или неподдерживаемая архитектура: {architecture}")

    def run_model_on_samples(
        self,
        model_id: int,
        model_name: str,
        architecture: str,
        backbone: str,
        weights_rel_path: str,
        dataset_storage_dir: str,
        samples: List[Tuple[int, str, str]],
    ) -> Dict[str, Any]:
        """Прогон нейросетевой модели по сэмплам датасета с расчетом метрик глубины."""
        dataset_dir = self.root_dir / dataset_storage_dir

        try:
            model_instance = self._get_or_load_model(architecture, backbone, weights_rel_path)
        except Exception as e:
            return {
                "model_id": model_id,
                "model_name": model_name,
                "run_status": "FAILED",
                "error_message": f"Ошибка загрузки модели: {e}",
                "mean_rmse": 0.0,
                "mean_silog": 0.0,
                "mean_absrel": 0.0,
                "mean_delta_1": 0.0,
                "avg_inference_time_ms": 0.0,
                "sample_metrics": [],
            }

        crop = "nyu" if "nyu" in str(dataset_dir).lower() else "garg"
        max_d = 80.0 if "kitti" in str(dataset_dir).lower() else 10.0

        sample_results = []
        rmses, silogs, absrels, deltas, latencies = [], [], [], [], []

        for sample_idx, rgb_rel, gt_rel in samples:
            rgb_full = dataset_dir / rgb_rel
            gt_full = dataset_dir / gt_rel

            if not rgb_full.exists():
                return {
                    "model_id": model_id,
                    "model_name": model_name,
                    "run_status": "FAILED",
                    "error_message": f"Файл изображения не найден: {rgb_full}",
                    "mean_rmse": 0.0,
                    "mean_silog": 0.0,
                    "mean_absrel": 0.0,
                    "mean_delta_1": 0.0,
                    "avg_inference_time_ms": 0.0,
                    "sample_metrics": [],
                }

            gt_depth = load_depth_map(gt_full)
            if gt_depth is None:
                return {
                    "model_id": model_id,
                    "model_name": model_name,
                    "run_status": "FAILED",
                    "error_message": f"Файл GT карты глубины не найден или поврежден: {gt_full}",
                    "mean_rmse": 0.0,
                    "mean_silog": 0.0,
                    "mean_absrel": 0.0,
                    "mean_delta_1": 0.0,
                    "avg_inference_time_ms": 0.0,
                    "sample_metrics": [],
                }

            t0 = time.perf_counter()

            img = Image.open(rgb_full).convert("RGB")
            rgb_np = np.array(img).transpose((2, 0, 1)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb_np)

            with torch.no_grad():
                pred_tensor = model_instance.predict(tensor)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            gt_tensor = torch.from_numpy(gt_depth)
            m_raw = compute_depth_metrics(pred_tensor, gt_tensor, crop_type=crop, max_depth=max_d)

            metrics = {
                "rmse": float(m_raw["RMSE"]),
                "silog": float(m_raw["SILog"]),
                "abs_rel": float(m_raw["AbsRel"]),
                "delta_1": float(m_raw["delta1"]),
            }

            sample_results.append({
                "sample_index": sample_idx,
                "rmse": metrics["rmse"],
                "silog": metrics["silog"],
                "abs_rel": metrics["abs_rel"],
                "delta_1": metrics["delta_1"],
                "inference_time_ms": infer_ms,
            })

            rmses.append(metrics["rmse"])
            silogs.append(metrics["silog"])
            absrels.append(metrics["abs_rel"])
            deltas.append(metrics["delta_1"])
            latencies.append(infer_ms)

        mean_rmse = float(np.mean(rmses)) if rmses else 0.0
        mean_silog = float(np.mean(silogs)) if silogs else 0.0
        mean_absrel = float(np.mean(absrels)) if absrels else 0.0
        mean_delta_1 = float(np.mean(deltas)) if deltas else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        return {
            "model_id": model_id,
            "model_name": model_name,
            "run_status": "COMPLETED",
            "error_message": "",
            "mean_rmse": mean_rmse,
            "mean_silog": mean_silog,
            "mean_absrel": mean_absrel,
            "mean_delta_1": mean_delta_1,
            "avg_inference_time_ms": avg_latency,
            "sample_metrics": sample_results,
        }

    def predict_single_image(
        self,
        architecture: str,
        backbone: str,
        weights_rel_path: str,
        image_bytes: bytes,
        colorize: bool = True,
        colormap: str = "magma",
    ) -> Dict[str, Any]:
        """Инференс произвольного изображения с получением карты глубины."""
        if not image_bytes:
            return {
                "success": False,
                "error_message": "Файл изображения пуст",
                "depth_png_data": b"",
                "width": 0,
                "height": 0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        if len(image_bytes) > 25 * 1024 * 1024:
            return {
                "success": False,
                "error_message": "Размер файла превышает допустимый лимит 25 МБ",
                "depth_png_data": b"",
                "width": 0,
                "height": 0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Не удалось декодировать изображение: {e}",
                "depth_png_data": b"",
                "width": 0,
                "height": 0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        orig_w, orig_h = img.size
        if orig_w < 16 or orig_h < 16:
            return {
                "success": False,
                "error_message": f"Разрешение изображения слишком мало: {orig_w}x{orig_h} (минимум 16x16)",
                "depth_png_data": b"",
                "width": orig_w,
                "height": orig_h,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        if orig_w > 4096 or orig_h > 4096:
            return {
                "success": False,
                "error_message": f"Разрешение изображения слишком велико: {orig_w}x{orig_h} (максимум 4096x4096)",
                "depth_png_data": b"",
                "width": orig_w,
                "height": orig_h,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        try:
            model_instance = self._get_or_load_model(architecture, backbone, weights_rel_path)
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Ошибка загрузки модели: {e}",
                "depth_png_data": b"",
                "width": orig_w,
                "height": orig_h,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        if model_instance is None:
            return {
                "success": False,
                "error_message": f"Модель {architecture} ({backbone}) не поддерживается",
                "depth_png_data": b"",
                "width": orig_w,
                "height": orig_h,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "inference_time_ms": 0.0,
            }

        img_rgb = img.convert("RGB")
        rgb_np = np.array(img_rgb).transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb_np).unsqueeze(0)

        # Паддинг до кратности 32 для Swin-Transformer backbones
        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor_input = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            tensor_input = tensor

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_tensor = model_instance.predict(tensor_input)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Кроп обратно до оригинального разрешения изображения
        pred_tensor = pred_tensor[..., :orig_h, :orig_w]
        depth_np = pred_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        min_d = float(np.min(depth_np))
        max_d = float(np.max(depth_np))

        buf = io.BytesIO()
        if colorize:
            norm = (depth_np - min_d) / (max_d - min_d + 1e-8)
            try:
                import matplotlib
                cmap = matplotlib.colormaps.get_cmap(colormap or "magma")
            except Exception:
                import matplotlib
                cmap = matplotlib.colormaps.get_cmap("magma")
            colored_rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
            out_img = Image.fromarray(colored_rgb)
        else:
            depth_mm = np.clip(depth_np * 1000.0, 0, 65535).astype(np.uint16)
            out_img = Image.fromarray(depth_mm)

        out_img.save(buf, format="PNG")
        depth_png = buf.getvalue()

        return {
            "success": True,
            "error_message": "",
            "depth_png_data": depth_png,
            "width": orig_w,
            "height": orig_h,
            "min_depth": min_d,
            "max_depth": max_d,
            "inference_time_ms": infer_ms,
        }
