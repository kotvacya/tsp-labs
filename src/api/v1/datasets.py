from fastapi import APIRouter, Depends, Query, status
from src.api.deps import get_dataset_service, require_role
from src.models import User
from src.schemas.dataset import DatasetCreateRequest, DatasetResponse
from src.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> list[DatasetResponse]:
    datasets = await dataset_service.list_datasets(limit=limit, offset=offset)
    return [DatasetResponse.model_validate(d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    dataset = await dataset_service.get_dataset(dataset_id)
    return DatasetResponse.model_validate(dataset)


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    req: DatasetCreateRequest,
    operator: User = Depends(require_role("OPERATOR", "ADMIN")),
    dataset_service: DatasetService = Depends(get_dataset_service),
) -> DatasetResponse:
    dataset = await dataset_service.create_dataset(user_id=operator.id, req=req)
    return DatasetResponse.model_validate(dataset)
