"""Расчёт метрик точности монокулярной оценки глубины (MDE)."""

from typing import Dict, Optional
import torch


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    eigen_crop: bool = True,
    valid_mask: Optional[torch.Tensor] = None,
    crop_type: Optional[str] = None,
) -> Dict[str, float]:
    """Вычисляет стандартный набор метрик глубины с учётом маски пустых пикселей.

    Args:
        pred: предсказанная глубина [H, W], [1, H, W] или [1, 1, H, W]
        gt: ground truth глубина [H, W], [1, H, W] или [1, 1, H, W]
        min_depth: минимальная допустимая валидная глубина в метрах
        max_depth: максимальная допустимая валидная глубина в метрах
        eigen_crop: применять ли стандартную оценку по обрезанной области (Eigen / Garg)
        valid_mask: внешняя булева маска непустых валидных пикселей [H, W] (например, от LiDAR)
        crop_type: принудительный выбор типа обрезки: 'garg', 'eigen', 'nyu' или 'none'

    Returns:
        Словарь с метриками: RMSE, SILog, AbsRel, SqRel, delta1, delta2, delta3,
        RMSE_log, log10, valid_pixels, empty_pixels
    """
    # Приведение тензоров к 2D [H, W]
    while pred.dim() > 2:
        pred = pred.squeeze(0)
    while gt.dim() > 2:
        gt = gt.squeeze(0)

    # Приведение к одному типу и устройству
    pred = pred.to(dtype=torch.float32)
    gt = gt.to(device=pred.device, dtype=torch.float32)

    # 1. Базовая маска валидных значений GT (исключение 0, отрицательных, inf и nan)
    mask = (gt > min_depth) & (gt < max_depth) & (~torch.isnan(gt)) & (~torch.isinf(gt))

    # 2. Учёт внешней маски валидных / непустых пикселей сенсора (если передана)
    if valid_mask is not None:
        while valid_mask.dim() > 2:
            valid_mask = valid_mask.squeeze(0)
        mask = mask & valid_mask.to(device=gt.device, dtype=torch.bool)

    # 3. Применение стандартных областей оценки (Evaluation Crops)
    H, W = gt.shape
    selected_crop = crop_type.lower() if crop_type else None

    # Garg crop для KITTI (Garg et al., ECCV 2016):
    # Y: [0.4081 * H : 0.9919 * H], X: [0.0359 * W : 0.9641 * W]
    if selected_crop == 'garg' or (selected_crop is None and eigen_crop and (max_depth > 20.0 or W > 1000)):
        crop_mask = torch.zeros_like(mask, dtype=torch.bool)
        top = int(0.40810811 * H)
        bottom = int(0.99189189 * H)
        left = int(0.03594771 * W)
        right = int(0.96405229 * W)
        crop_mask[top:bottom, left:right] = True
        mask = mask & crop_mask

    # Eigen crop для KITTI (Eigen et al., NIPS 2014)
    elif selected_crop == 'eigen' and (max_depth > 20.0 or W > 1000):
        crop_mask = torch.zeros_like(mask, dtype=torch.bool)
        top = int(0.3324324 * H)
        bottom = int(0.91351351 * H)
        left = int(0.0359477 * W)
        right = int(0.96405229 * W)
        crop_mask[top:bottom, left:right] = True
        mask = mask & crop_mask

    # Eigen crop для NYU Depth v2 (480x640: [45:471, 41:601])
    elif selected_crop == 'nyu' or (eigen_crop and H == 480 and W == 640):
        crop_mask = torch.zeros_like(mask, dtype=torch.bool)
        crop_mask[45:471, 41:601] = True
        mask = mask & crop_mask

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    # Подсчёт пустых пикселей (gt <= 0 или NaN)
    empty_pixels_count = int(((gt <= 0.0) | torch.isnan(gt)).sum().item())

    if pred_valid.numel() == 0:
        return {
            'RMSE': float('nan'),
            'SILog': float('nan'),
            'AbsRel': float('nan'),
            'SqRel': float('nan'),
            'delta1': float('nan'),
            'delta2': float('nan'),
            'delta3': float('nan'),
            'RMSE_log': float('nan'),
            'log10': float('nan'),
            'valid_pixels': 0,
            'empty_pixels': empty_pixels_count,
        }

    # Ограничение предсказаний диапазоном валидности
    pred_valid = torch.clamp(pred_valid, min=min_depth, max=max_depth)

    # 1. RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))

    # 2. SILog (Scale-Invariant Logarithmic Error, в процентах)
    log_diff = torch.log(pred_valid) - torch.log(gt_valid)
    silog = torch.sqrt(torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2)) * 100.0

    # 3. AbsRel (Absolute Relative Difference)
    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid)

    # 4. SqRel (Squared Relative Difference)
    sq_rel = torch.mean(((pred_valid - gt_valid) ** 2) / gt_valid)

    # 5. Threshold metrics (delta_i = % пикселей, где max(p/g, g/p) < 1.25^i)
    ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()

    # 6. RMSE_log & Log10
    rmse_log = torch.sqrt(torch.mean(log_diff ** 2))
    log10_err = torch.mean(torch.abs(torch.log10(pred_valid) - torch.log10(gt_valid)))

    return {
        'RMSE': float(rmse.item()),
        'SILog': float(silog.item()),
        'AbsRel': float(abs_rel.item()),
        'SqRel': float(sq_rel.item()),
        'delta1': float(delta1.item()),
        'delta2': float(delta2.item()),
        'delta3': float(delta3.item()),
        'RMSE_log': float(rmse_log.item()),
        'log10': float(log10_err.item()),
        'valid_pixels': int(pred_valid.numel()),
        'empty_pixels': empty_pixels_count,
    }
