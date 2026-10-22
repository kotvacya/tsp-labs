from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models import BenchmarkExperiment, ExperimentModelRun
from src.repositories.base import BaseRepository


class BenchmarkExperimentRepository(BaseRepository[BenchmarkExperiment]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, BenchmarkExperiment)

    async def create_with_models(
        self, experiment: BenchmarkExperiment, model_ids: list[int]
    ) -> BenchmarkExperiment:
        self.db.add(experiment)
        await self.db.flush()

        for mid in model_ids:
            run = ExperimentModelRun(experiment_id=experiment.id, model_id=mid)
            self.db.add(run)

        await self.db.flush()
        result = await self.get_experiment_with_models(experiment.id)
        assert result is not None
        return result

    async def get_experiment_with_models(
        self, experiment_id: int
    ) -> BenchmarkExperiment | None:
        stmt = (
            select(BenchmarkExperiment)
            .options(
                selectinload(BenchmarkExperiment.model_runs).selectinload(
                    ExperimentModelRun.model
                )
            )
            .where(BenchmarkExperiment.id == experiment_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self, user_id: int, limit: int = 100, offset: int = 0
    ) -> Sequence[BenchmarkExperiment]:
        stmt = (
            select(BenchmarkExperiment)
            .where(BenchmarkExperiment.user_id == user_id)
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()
