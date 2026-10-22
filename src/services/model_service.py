from typing import Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import AlreadyExistsException, NotFoundException
from src.models import NeuralModel
from src.repositories.model_repo import NeuralModelRepository
from src.schemas.model import NeuralModelCreateRequest, NeuralModelUpdateRequest


class NeuralModelService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_repo = NeuralModelRepository(db)

    async def create_model(self, req: NeuralModelCreateRequest) -> NeuralModel:
        existing = await self.model_repo.get_by_name(req.name)
        if existing:
            raise AlreadyExistsException(f"Model with name '{req.name}' already exists")

        model = NeuralModel(
            name=req.name,
            architecture=req.architecture,
            backbone=req.backbone,
            weights_rel_path=req.weights_rel_path,
            is_active=req.is_active,
        )
        return await self.model_repo.create(model)

    async def get_model(self, model_id: int) -> NeuralModel:
        model = await self.model_repo.get_by_id(model_id)
        if not model:
            raise NotFoundException(f"Neural model with id={model_id} not found")
        return model

    async def list_models(
        self, is_active: bool | None = None, limit: int = 100, offset: int = 0
    ) -> Sequence[NeuralModel]:
        return await self.model_repo.list_models(is_active=is_active, limit=limit, offset=offset)

    async def update_model(self, model_id: int, req: NeuralModelUpdateRequest) -> NeuralModel:
        model = await self.get_model(model_id)
        if req.name is not None and req.name != model.name:
            existing = await self.model_repo.get_by_name(req.name)
            if existing:
                raise AlreadyExistsException(f"Model with name '{req.name}' already exists")
            model.name = req.name

        if req.architecture is not None:
            model.architecture = req.architecture
        if req.backbone is not None:
            model.backbone = req.backbone
        if req.weights_rel_path is not None:
            model.weights_rel_path = req.weights_rel_path
        if req.is_active is not None:
            model.is_active = req.is_active

        await self.db.flush()
        await self.db.refresh(model)
        return model

    async def delete_model(self, model_id: int) -> None:
        model = await self.get_model(model_id)
        await self.model_repo.delete(model)

    async def get_model_with_experiments(self, model_id: int) -> NeuralModel:
        model = await self.model_repo.get_model_with_experiments(model_id)
        if not model:
            raise NotFoundException(f"Neural model with id={model_id} not found")
        return model
