import os
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import hash_password
from src.models.entities import Dataset, DatasetSample, NeuralModel, User


async def seed_initial_data(session: AsyncSession) -> None:
    """Инициализация базы данных начальными пользователями, датасетами, сэмплами и моделями."""

    # 1. Пользователи
    admin = await session.scalar(select(User).where(User.email == "admin@mdebench.com"))
    if not admin:
        admin = User(
            email="admin@mdebench.com",
            hashed_password=hash_password("adminpass"),
            full_name="System Administrator",
            role="ADMIN",
        )
        session.add(admin)

    operator = await session.scalar(select(User).where(User.email == "operator@mdebench.com"))
    if not operator:
        operator = User(
            email="operator@mdebench.com",
            hashed_password=hash_password("operatorpass"),
            full_name="Data Operator",
            role="OPERATOR",
        )
        session.add(operator)

    researcher = await session.scalar(select(User).where(User.email == "researcher@mdebench.com"))
    if not researcher:
        researcher = User(
            email="researcher@mdebench.com",
            hashed_password=hash_password("researcherpass"),
            full_name="Lead Researcher",
            role="RESEARCHER",
        )
        session.add(researcher)

    await session.flush()

    # 2. Нейросетевые модели (IEBins и UniDepth)
    models_to_seed = [
        {
            "name": "IEBins-Tiny-NYU",
            "architecture": "IEBins",
            "backbone": "Swin-Tiny",
            "weights_rel_path": "checkpoints/iebins/nyu_T.pth",
            "is_active": True,
        },
        {
            "name": "IEBins-Large-NYU",
            "architecture": "IEBins",
            "backbone": "Swin-Large",
            "weights_rel_path": "checkpoints/iebins/nyu_L.pth",
            "is_active": True,
        },
        {
            "name": "IEBins-Tiny-KITTI",
            "architecture": "IEBins",
            "backbone": "Swin-Tiny",
            "weights_rel_path": "checkpoints/iebins/kittieigen_T.pth",
            "is_active": True,
        },
        {
            "name": "IEBins-Large-KITTI",
            "architecture": "IEBins",
            "backbone": "Swin-Large",
            "weights_rel_path": "checkpoints/iebins/kittieigen_L.pth",
            "is_active": True,
        },
        {
            "name": "UniDepth-v2-ViTL14",
            "architecture": "UniDepth",
            "backbone": "ViT-L/14",
            "weights_rel_path": "checkpoints/unidepth/unidepth-v2-vitl14",
            "is_active": True,
        },
    ]

    for m_data in models_to_seed:
        existing_m = await session.scalar(select(NeuralModel).where(NeuralModel.name == m_data["name"]))
        if not existing_m:
            model = NeuralModel(**m_data)
            session.add(model)

    await session.flush()

    # 3. Датасет NYU Depth v2
    root_dir = Path(__file__).resolve().parent.parent.parent
    nyu_dataset = await session.scalar(select(Dataset).where(Dataset.name == "NYU-Depth-V2"))
    if not nyu_dataset:
        nyu_dataset = Dataset(
            user_id=operator.id,
            name="NYU-Depth-V2",
            description="NYU Depth v2 indoor benchmark dataset (1449 labeled pairs, 654 test split)",
            storage_dir="data/nyu_depth_v2",
            total_samples=0,
        )
        session.add(nyu_dataset)
        await session.flush()

        # Заполняем сэмплы из extracted
        split_file = root_dir / "data" / "splits" / "nyu_test_indices.txt"
        samples_to_add = []
        if split_file.exists():
            with open(split_file, "r") as f:
                indices = [int(line.strip()) for line in f if line.strip()]
            for s_idx, idx in enumerate(indices):
                samples_to_add.append(
                    DatasetSample(
                        dataset_id=nyu_dataset.id,
                        sample_index=s_idx,
                        rgb_image_rel_path=f"extracted/rgb/sample_{idx:04d}.png",
                        gt_depth_rel_path=f"extracted/depth/sample_{idx:04d}.png",
                    )
                )
            nyu_dataset.total_samples = len(samples_to_add)
            session.add_all(samples_to_add)

    # 4. Датасет KITTI Eigen Split
    kitti_dataset = await session.scalar(select(Dataset).where(Dataset.name == "KITTI-Eigen-Split"))
    if not kitti_dataset:
        kitti_dataset = Dataset(
            user_id=operator.id,
            name="KITTI-Eigen-Split",
            description="KITTI Outdoor Autonomous Driving Depth Benchmark (Eigen test split with LiDAR GT)",
            storage_dir="data/kitti_eigen_val",
            total_samples=0,
        )
        session.add(kitti_dataset)
        await session.flush()

        # Заполняем сэмплы из файла сплита
        kitti_split = root_dir / "data" / "splits" / "eigen_test_files_with_gt.txt"
        kitti_root = root_dir / "data" / "kitti_eigen_val"
        kitti_samples = []
        if kitti_split.exists() and kitti_root.exists():
            with open(kitti_split, "r") as f:
                s_idx = 0
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] != "None":
                        rgb_rel = parts[0]
                        gt_rel = parts[1]
                        date_prefix = rgb_rel.split("/")[0]
                        if not gt_rel.startswith(date_prefix + "/"):
                            gt_rel = os.path.join(date_prefix, gt_rel)

                        rgb_full = kitti_root / rgb_rel
                        gt_full = kitti_root / gt_rel
                        if rgb_full.exists() and gt_full.exists():
                            kitti_samples.append(
                                DatasetSample(
                                    dataset_id=kitti_dataset.id,
                                    sample_index=s_idx,
                                    rgb_image_rel_path=rgb_rel,
                                    gt_depth_rel_path=gt_rel,
                                )
                            )
                            s_idx += 1
            kitti_dataset.total_samples = len(kitti_samples)
            session.add_all(kitti_samples)

    await session.commit()
