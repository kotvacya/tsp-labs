from fastapi import APIRouter, Depends, Query, status
from src.api.deps import get_benchmark_service, get_current_user
from src.core.grpc_client import get_grpc_client
from src.models import User
from src.schemas.experiment import (
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentListItemResponse,
    LeaderboardItemResponse,
)
from src.services.experiment_service import BenchmarkExperimentService

router = APIRouter(prefix="/experiments", tags=["Benchmark Experiments"])


@router.get("/compute/status", tags=["gRPC Compute Engine"])
async def get_compute_status() -> dict[str, str]:
    """Проверка доступности и статуса удалённого gRPC сервиса вычислений."""
    client = get_grpc_client()
    return await client.health_check()


@router.get("", response_model=list[ExperimentListItemResponse])
async def list_experiments(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    benchmark_service: BenchmarkExperimentService = Depends(get_benchmark_service),
) -> list[ExperimentListItemResponse]:
    exps = await benchmark_service.list_experiments(limit=limit, offset=offset)
    return [ExperimentListItemResponse.model_validate(e) for e in exps]


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    experiment_id: int,
    benchmark_service: BenchmarkExperimentService = Depends(get_benchmark_service),
) -> ExperimentDetailResponse:
    exp = await benchmark_service.get_experiment(experiment_id)
    return ExperimentDetailResponse.model_validate(exp)


@router.get("/{experiment_id}/leaderboard", response_model=list[LeaderboardItemResponse])
async def get_experiment_leaderboard(
    experiment_id: int,
    sort_by: str = Query(
        default="rmse",
        description="Metric to rank by: rmse, silog, absrel, delta_1, time",
    ),
    order: str = Query(
        default="asc",
        description="Sorting direction: asc or desc",
    ),
    benchmark_service: BenchmarkExperimentService = Depends(get_benchmark_service),
) -> list[LeaderboardItemResponse]:
    return await benchmark_service.get_leaderboard(
        experiment_id=experiment_id, sort_by=sort_by, order=order
    )


@router.post("", response_model=ExperimentDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    req: ExperimentCreateRequest,
    current_user: User = Depends(get_current_user),
    benchmark_service: BenchmarkExperimentService = Depends(get_benchmark_service),
) -> ExperimentDetailResponse:
    exp = await benchmark_service.create_experiment(user_id=current_user.id, req=req)
    return ExperimentDetailResponse.model_validate(exp)


@router.post("/{experiment_id}/run", response_model=ExperimentDetailResponse)
async def run_experiment(
    experiment_id: int,
    max_samples: int = Query(default=20, ge=1, le=1000, description="Max samples to evaluate"),
    current_user: User = Depends(get_current_user),
    benchmark_service: BenchmarkExperimentService = Depends(get_benchmark_service),
) -> ExperimentDetailResponse:
    """Запуск вычисления бенчмарка моделей на датасете через gRPC Compute Service."""
    exp = await benchmark_service.run_experiment(experiment_id=experiment_id, max_samples=max_samples)
    return ExperimentDetailResponse.model_validate(exp)
