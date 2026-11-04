from fastapi import APIRouter, Depends, Query, status
from src.api.deps import get_current_user, get_model_service, require_role
from src.models import User
from src.schemas.model import (
    NeuralModelCreateRequest,
    NeuralModelResponse,
    NeuralModelUpdateRequest,
)
from src.services.model_service import NeuralModelService

router = APIRouter(prefix="/models", tags=["Neural Models"])


@router.get("", response_model=list[NeuralModelResponse])
async def list_models(
    is_active: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model_service: NeuralModelService = Depends(get_model_service),
) -> list[NeuralModelResponse]:
    models = await model_service.list_models(is_active=is_active, limit=limit, offset=offset)
    return [NeuralModelResponse.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=NeuralModelResponse)
async def get_model(
    model_id: int,
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.get_model(model_id)
    return NeuralModelResponse.model_validate(model)


@router.post("", response_model=NeuralModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    req: NeuralModelCreateRequest,
    current_user: User = Depends(get_current_user),
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.create_model(req)
    return NeuralModelResponse.model_validate(model)


@router.patch("/{model_id}", response_model=NeuralModelResponse)
async def update_model(
    model_id: int,
    req: NeuralModelUpdateRequest,
    current_user: User = Depends(get_current_user),
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.update_model(model_id, req)
    return NeuralModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    admin_user: User = Depends(require_role("ADMIN")),
    model_service: NeuralModelService = Depends(get_model_service),
) -> None:
    await model_service.delete_model(model_id)
