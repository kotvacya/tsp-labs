from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models import NeuralModel, ExperimentModelRun
from src.repositories.base import BaseRepository


class NeuralModelRepository(BaseRepository[NeuralModel]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, NeuralModel)

    async def get_by_name(self, name: str) -> NeuralModel | None:
        stmt = select(NeuralModel).where(NeuralModel.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_models(
        self, is_active: bool | None = None, limit: int = 100, offset: int = 0
    ) -> Sequence[NeuralModel]:
        stmt = select(NeuralModel)
        if is_active is not None:
            stmt = stmt.where(NeuralModel.is_active == is_active)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_model_with_experiments(self, model_id: int) -> NeuralModel | None:
        stmt = (
            select(NeuralModel)
            .options(
                selectinload(NeuralModel.runs).selectinload(ExperimentModelRun.experiment)
            )
            .where(NeuralModel.id == model_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
