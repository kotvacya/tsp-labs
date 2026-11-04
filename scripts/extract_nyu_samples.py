#!/usr/bin/env python3
"""Извлечение тестовых RGB-изображений и карт глубин из nyu_depth_v2_labeled.mat в PNG."""

import os
import sys
import time
from pathlib import Path
import numpy as np

try:
    import h5py
    from PIL import Image
except ImportError:
    print("Требуются h5py и Pillow. Запустите через python с установленными зависимостями.")
    sys.exit(1)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    mat_path = root_dir / "data" / "nyu_depth_v2" / "nyu_depth_v2_labeled.mat"
    split_file = root_dir / "data" / "splits" / "nyu_test_indices.txt"

    out_rgb = root_dir / "data" / "nyu_depth_v2" / "extracted" / "rgb"
    out_depth = root_dir / "data" / "nyu_depth_v2" / "extracted" / "depth"
    out_depth_vis = root_dir / "data" / "nyu_depth_v2" / "extracted" / "depth_vis"

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    out_depth_vis.mkdir(parents=True, exist_ok=True)

    if not mat_path.exists():
        print(f"Файл {mat_path} не найден!")
        sys.exit(1)

    with open(split_file, "r") as f:
        indices = [int(line.strip()) for line in f if line.strip()]

    print(f"Извлечение {len(indices)} сэмплов из {mat_path.name}...")
    t0 = time.time()

    with h5py.File(mat_path, "r") as mat:
        for i, idx in enumerate(indices):
            img_raw = mat["images"][idx]
            depth_raw = mat["depths"][idx]

            # MATLAB column-major (3, 640, 480) -> C row-major (480, 640, 3)
            rgb_np = np.transpose(img_raw, (2, 1, 0)).astype(np.uint8)
            depth_np = np.transpose(depth_raw, (1, 0)).astype(np.float32)

            rgb_file = out_rgb / f"sample_{idx:04d}.png"
            depth_file = out_depth / f"sample_{idx:04d}.png"
            vis_file = out_depth_vis / f"sample_{idx:04d}.png"

            if not rgb_file.exists():
                Image.fromarray(rgb_np).save(rgb_file)

            if not depth_file.exists():
                depth_mm = (depth_np * 1000.0).astype(np.uint16)
                Image.fromarray(depth_mm).save(depth_file)

            if not vis_file.exists():
                norm = np.clip((depth_np - 0.5) / (8.0 - 0.5), 0, 1)
                gray = (norm * 255).astype(np.uint8)
                Image.fromarray(gray).save(vis_file)

            if (i + 1) % 100 == 0 or (i + 1) == len(indices):
                print(f"[{i + 1}/{len(indices)}] сэмплов извлечено...")

    elapsed = time.time() - t0
    print(f"Готово за {elapsed:.1f} сек. Файлы сохранены в {out_rgb.parent}")


if __name__ == "__main__":
    main()
