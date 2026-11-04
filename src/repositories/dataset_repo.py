from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models import Dataset
from src.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    def __init__(self, db: AsyncSession):
        super().__init__(db, Dataset)

    async def get_by_name(self, name: str) -> Dataset | None:
        stmt = select(Dataset).where(Dataset.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
