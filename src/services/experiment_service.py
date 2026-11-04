from typing import Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import BadRequestException, NotFoundException
from src.models import BenchmarkExperiment
from src.repositories.dataset_repo import DatasetRepository
from src.repositories.experiment_repo import BenchmarkExperimentRepository
from src.repositories.model_repo import NeuralModelRepository
from src.schemas.experiment import ExperimentCreateRequest, LeaderboardItemResponse


class BenchmarkExperimentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.experiment_repo = BenchmarkExperimentRepository(db)
        self.dataset_repo = DatasetRepository(db)
        self.model_repo = NeuralModelRepository(db)

    async def create_experiment(
        self, user_id: int, req: ExperimentCreateRequest
    ) -> BenchmarkExperiment:
        dataset = await self.dataset_repo.get_by_id(req.dataset_id)
        if not dataset:
            raise NotFoundException(f"Dataset with id={req.dataset_id} not found")

        for mid in req.model_ids:
            model = await self.model_repo.get_by_id(mid)
            if not model:
                raise NotFoundException(f"Neural model with id={mid} not found")
            if not model.is_active:
                raise BadRequestException(
                    f"Neural model '{model.name}' (id={mid}) is inactive and cannot be benchmarked"
                )

        experiment = BenchmarkExperiment(
            user_id=user_id,
            dataset_id=req.dataset_id,
            title=req.title,
            status="PENDING",
        )
        return await self.experiment_repo.create_with_models(experiment, req.model_ids)

    async def get_experiment(self, experiment_id: int) -> BenchmarkExperiment:
        experiment = await self.experiment_repo.get_experiment_with_models(experiment_id)
        if not experiment:
            raise NotFoundException(f"Benchmark experiment with id={experiment_id} not found")
        return experiment

    async def list_experiments(
        self, user_id: int | None = None, limit: int = 100, offset: int = 0
    ) -> Sequence[BenchmarkExperiment]:
        if user_id is not None:
            return await self.experiment_repo.list_by_user(user_id=user_id, limit=limit, offset=offset)
        return await self.experiment_repo.list_all(limit=limit, offset=offset)

    async def get_leaderboard(
        self, experiment_id: int, sort_by: str = "rmse", order: str = "asc"
    ) -> list[LeaderboardItemResponse]:
        exp = await self.get_experiment(experiment_id)
        metric_map = {
            "rmse": "mean_rmse",
            "silog": "mean_silog",
            "absrel": "mean_absrel",
            "delta_1": "mean_delta_1",
            "time": "avg_inference_time_ms",
            "speed": "avg_inference_time_ms",
        }
        attr = metric_map.get(sort_by.lower(), "mean_rmse")
        if sort_by.lower() == "delta_1":
            reverse = (order.lower() != "asc")
        else:
            reverse = (order.lower() == "desc")

        def sort_key(run):
            val = getattr(run, attr, None)
            if val is None:
                return float("inf") if not reverse else float("-inf")
            return val

        sorted_runs = sorted(exp.model_runs, key=sort_key, reverse=reverse)

        return [
            LeaderboardItemResponse(
                rank=idx + 1,
                model_id=run.model_id,
                model_name=run.model.name if run.model else f"Model #{run.model_id}",
                architecture=run.model.architecture if run.model else "unknown",
                backbone=run.model.backbone if run.model else "unknown",
                mean_rmse=run.mean_rmse,
                mean_silog=run.mean_silog,
                mean_absrel=run.mean_absrel,
                mean_delta_1=run.mean_delta_1,
                avg_inference_time_ms=run.avg_inference_time_ms,
                run_status=run.run_status,
            )
            for idx, run in enumerate(sorted_runs)
        ]
