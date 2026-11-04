from typing import Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import AlreadyExistsException, NotFoundException
from src.models import Dataset
from src.repositories.dataset_repo import DatasetRepository
from src.schemas.dataset import DatasetCreateRequest


class DatasetService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.dataset_repo = DatasetRepository(db)

    async def create_dataset(self, user_id: int, req: DatasetCreateRequest) -> Dataset:
        existing = await self.dataset_repo.get_by_name(req.name)
        if existing:
            raise AlreadyExistsException(f"Dataset with name '{req.name}' already exists")

        dataset = Dataset(
            user_id=user_id,
            name=req.name,
            description=req.description,
            storage_dir=req.storage_dir,
            total_samples=req.total_samples,
        )
        return await self.dataset_repo.create(dataset)

    async def get_dataset(self, dataset_id: int) -> Dataset:
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise NotFoundException(f"Dataset with id={dataset_id} not found")
        return dataset

    async def list_datasets(self, limit: int = 100, offset: int = 0) -> Sequence[Dataset]:
        return await self.dataset_repo.list_all(limit=limit, offset=offset)
