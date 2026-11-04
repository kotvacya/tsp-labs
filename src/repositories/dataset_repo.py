from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.models import Dataset
from src.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, Dataset)

    async def get_by_name(self, name: str) -> Dataset | None:
        stmt = select(Dataset).where(Dataset.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_samples(self, dataset_id: int) -> Dataset | None:
        stmt = (
            select(Dataset)
            .options(selectinload(Dataset.samples))
            .where(Dataset.id == dataset_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
