from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from src import models

async def create_user(db: AsyncSession, email: str, password: str, name: str, role: str) -> models.User:
    u = models.User(email=email, hashed_password=password, full_name=name, role=role)
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return u

async def get_user(db: AsyncSession, user_id: int) -> models.User | None:
    return await db.get(models.User, user_id)

async def create_model(db: AsyncSession, name: str, arch: str, backbone: str, weights: str) -> models.NeuralModel:
    m = models.NeuralModel(name=name, architecture=arch, backbone=backbone, weights_rel_path=weights)
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return m

async def get_model(db: AsyncSession, model_id: int) -> models.NeuralModel | None:
    return await db.get(models.NeuralModel, model_id)

async def update_model(db: AsyncSession, model_id: int, is_active: bool) -> models.NeuralModel | None:
    m = await db.get(models.NeuralModel, model_id)
    if m:
        m.is_active = is_active
        await db.commit()
        await db.refresh(m)
    return m

async def delete_model(db: AsyncSession, model_id: int) -> bool:
    m = await db.get(models.NeuralModel, model_id)
    if m:
        await db.delete(m)
        await db.commit()
        return True
    return False

async def create_dataset(db: AsyncSession, user_id: int, name: str, storage_dir: str) -> models.Dataset:
    d = models.Dataset(user_id=user_id, name=name, storage_dir=storage_dir)
    db.add(d)
    await db.commit()
    await db.refresh(d)
    return d

async def create_experiment(db: AsyncSession, user_id: int, dataset_id: int, title: str, model_ids: list[int]) -> models.BenchmarkExperiment:
    exp = models.BenchmarkExperiment(user_id=user_id, dataset_id=dataset_id, title=title)
    db.add(exp)
    await db.flush()
    for mid in model_ids:
        run = models.ExperimentModelRun(experiment_id=exp.id, model_id=mid)
        db.add(run)
    await db.commit()
    await db.refresh(exp)
    return exp

async def get_experiment_with_models(db: AsyncSession, experiment_id: int) -> models.BenchmarkExperiment | None:
    stmt = (
        select(models.BenchmarkExperiment)
        .options(
            selectinload(models.BenchmarkExperiment.model_runs)
            .selectinload(models.ExperimentModelRun.model)
        )
        .where(models.BenchmarkExperiment.id == experiment_id)
    )
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

async def get_model_with_experiments(db: AsyncSession, model_id: int) -> models.NeuralModel | None:
    stmt = (
        select(models.NeuralModel)
        .options(
            selectinload(models.NeuralModel.runs)
            .selectinload(models.ExperimentModelRun.experiment)
        )
        .where(models.NeuralModel.id == model_id)
    )
    res = await db.execute(stmt)
    return res.scalar_one_or_none()
