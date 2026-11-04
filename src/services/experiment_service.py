from datetime import datetime, timezone
from typing import Sequence, Optional
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import BadRequestException, NotFoundException
from src.core.grpc_client import GrpcComputeClient, get_grpc_client
from src.models import BenchmarkExperiment, MetricRecord
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

    async def run_experiment(
        self,
        experiment_id: int,
        max_samples: int = 20,
        grpc_client: Optional[GrpcComputeClient] = None,
    ) -> BenchmarkExperiment:
        client = grpc_client or get_grpc_client()
        exp = await self.get_experiment(experiment_id)
        if not exp:
            raise NotFoundException(f"Benchmark experiment with id={experiment_id} not found")

        if exp.status == "RUNNING":
            raise BadRequestException(f"Эксперимент #{experiment_id} уже выполняется в данный момент")

        dataset = await self.dataset_repo.get_with_samples(exp.dataset_id)
        if not dataset:
            raise NotFoundException(f"Dataset with id={exp.dataset_id} not found")

        # Очистка старых покадровых метрик перед повторным запуском
        run_ids = [r.id for r in exp.model_runs]
        if run_ids:
            await self.db.execute(delete(MetricRecord).where(MetricRecord.run_id.in_(run_ids)))

        # Сброс агрегированных метрик для запускаемых прогонов
        for r in exp.model_runs:
            r.run_status = "RUNNING"
            r.mean_rmse = None
            r.mean_silog = None
            r.mean_absrel = None
            r.mean_delta_1 = None
            r.avg_inference_time_ms = None

        exp.status = "RUNNING"
        exp.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        exp.finished_at = None
        await self.db.flush()

        models_payload = [
            {
                "id": run.model.id,
                "name": run.model.name,
                "architecture": run.model.architecture,
                "backbone": run.model.backbone,
                "weights_rel_path": run.model.weights_rel_path,
            }
            for run in exp.model_runs
            if run.model
        ]

        samples_payload = [
            {
                "sample_index": s.sample_index,
                "rgb_image_rel_path": s.rgb_image_rel_path,
                "gt_depth_rel_path": s.gt_depth_rel_path,
            }
            for s in dataset.samples
        ]

        try:
            response = await client.execute_benchmark(
                experiment_id=exp.id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                dataset_storage_dir=dataset.storage_dir,
                models=models_payload,
                samples=samples_payload,
                max_samples=max_samples,
            )
        except Exception as e:
            exp.status = "FAILED"
            exp.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            for r in exp.model_runs:
                if r.run_status == "RUNNING":
                    r.run_status = "FAILED"
            await self.db.commit()
            raise e

        sample_id_by_index = {s.sample_index: s.id for s in dataset.samples}

        for model_res in response.results:
            run = next((r for r in exp.model_runs if r.model_id == model_res.model_id), None)
            if run:
                run.run_status = model_res.run_status
                run.mean_rmse = model_res.mean_rmse
                run.mean_silog = model_res.mean_silog
                run.mean_absrel = model_res.mean_absrel
                run.mean_delta_1 = model_res.mean_delta_1
                run.avg_inference_time_ms = model_res.avg_inference_time_ms

                for sm in model_res.sample_metrics:
                    sample_db_id = sample_id_by_index.get(sm.sample_index)
                    if sample_db_id:
                        record = MetricRecord(
                            run_id=run.id,
                            sample_id=sample_db_id,
                            silog=sm.silog,
                            abs_rel=sm.abs_rel,
                            rmse=sm.rmse,
                            delta_1=sm.delta_1,
                            inference_time_ms=sm.inference_time_ms,
                        )
                        self.db.add(record)

        all_failed = all(r.run_status == "FAILED" for r in exp.model_runs) if exp.model_runs else False
        exp.status = "FAILED" if all_failed else "FINISHED"
        exp.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()

        return await self.get_experiment(experiment_id)
