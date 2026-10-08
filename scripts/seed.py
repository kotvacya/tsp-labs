import asyncio
from sqlalchemy import select
from src.database import AsyncSessionLocal
from src.init_db import init_db
from src import models

async def seed():
    await init_db()
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(models.User))
        if res.scalars().first():
            return

        u1 = models.User(email="admin@test.local", hashed_password="pwd", full_name="Admin", role="ADMIN")
        u2 = models.User(email="user@test.local", hashed_password="pwd", full_name="User", role="RESEARCHER")
        db.add_all([u1, u2])
        await db.commit()

        d = models.Dataset(user_id=u1.id, name="kitti_val", storage_dir="/data/kitti", total_samples=2)
        db.add(d)
        await db.commit()

        s1 = models.DatasetSample(dataset_id=d.id, sample_index=1, rgb_image_rel_path="1.png", gt_depth_rel_path="1.npy")
        s2 = models.DatasetSample(dataset_id=d.id, sample_index=2, rgb_image_rel_path="2.png", gt_depth_rel_path="2.npy")
        db.add_all([s1, s2])

        m1 = models.NeuralModel(name="AdaBins", architecture="transformer", backbone="efficientnet", weights_rel_path="/w/adabins.pt")
        m2 = models.NeuralModel(name="DPT", architecture="dpt", backbone="vit", weights_rel_path="/w/dpt.pt")
        db.add_all([m1, m2])
        await db.commit()

        exp = models.BenchmarkExperiment(user_id=u2.id, dataset_id=d.id, title="Test Benchmark")
        db.add(exp)
        await db.flush()

        run1 = models.ExperimentModelRun(experiment_id=exp.id, model_id=m1.id, mean_rmse=0.25, mean_absrel=0.08)
        run2 = models.ExperimentModelRun(experiment_id=exp.id, model_id=m2.id, mean_rmse=0.21, mean_absrel=0.07)
        db.add_all([run1, run2])
        await db.commit()

if __name__ == "__main__":
    asyncio.run(seed())
    print("Seed completed.")
