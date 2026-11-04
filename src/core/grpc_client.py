"""Клиент gRPC для взаимодействия FastAPI с Benchmark Compute Service."""

from typing import List, Optional
import grpc

from src.config import settings
from src.core.exceptions import AppException
from src.grpc_gen import benchmark_pb2, benchmark_pb2_grpc


class GrpcComputeClient:
    """Асинхронный клиент gRPC сервиса вычислений."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or settings.GRPC_COMPUTE_HOST
        self.port = port or settings.GRPC_COMPUTE_PORT
        self.target = f"{self.host}:{self.port}"

    async def health_check(self) -> dict[str, str]:
        """Проверка доступности gRPC сервера вычислений."""
        try:
            async with grpc.aio.insecure_channel(self.target) as channel:
                stub = benchmark_pb2_grpc.BenchmarkComputeServiceStub(channel)
                resp = await stub.HealthCheck(benchmark_pb2.HealthRequest(), timeout=5.0)
                return {
                    "status": resp.status,
                    "device": resp.device,
                    "version": resp.version,
                    "target": self.target,
                }
        except grpc.RpcError as e:
            raise AppException(
                status_code=503,
                message=f"Compute Service unavailable on {self.target}: {e.details() or e.code().name}",
            )

    async def execute_benchmark(
        self,
        experiment_id: int,
        dataset_id: int,
        dataset_name: str,
        dataset_storage_dir: str,
        models: List[dict],
        samples: List[dict],
        max_samples: int = 20,
    ) -> benchmark_pb2.BenchmarkResponse:
        """Отправка задачи бенчмаркинга в gRPC вычислительный сервис."""
        proto_models = [
            benchmark_pb2.ModelExecutionConfig(
                model_id=m["id"],
                model_name=m["name"],
                architecture=m["architecture"],
                backbone=m["backbone"],
                weights_rel_path=m["weights_rel_path"],
            )
            for m in models
        ]

        proto_samples = [
            benchmark_pb2.SamplePayload(
                sample_index=s["sample_index"],
                rgb_rel_path=s["rgb_image_rel_path"],
                gt_depth_rel_path=s["gt_depth_rel_path"],
            )
            for s in samples
        ]

        req = benchmark_pb2.BenchmarkRequest(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_storage_dir=dataset_storage_dir,
            models=proto_models,
            samples=proto_samples,
            max_samples=max_samples,
        )

        try:
            async with grpc.aio.insecure_channel(self.target) as channel:
                stub = benchmark_pb2_grpc.BenchmarkComputeServiceStub(channel)
                # Таймаут на время выполнения вычислений
                resp = await stub.ExecuteBenchmark(req, timeout=120.0)
                return resp
        except grpc.RpcError as e:
            raise AppException(
                status_code=502,
                message=f"gRPC Compute Error ({e.code().name}): {e.details() or 'Remote error'}",
            )

    async def predict_single_image(
        self,
        model: dict,
        image_bytes: bytes,
        colorize: bool = True,
        colormap: str = "magma",
    ) -> benchmark_pb2.SingleImagePredictResponse:
        """Отправка одиночного изображения в gRPC сервис для инференса карты глубины."""
        proto_model = benchmark_pb2.ModelExecutionConfig(
            model_id=model["id"],
            model_name=model["name"],
            architecture=model["architecture"],
            backbone=model["backbone"],
            weights_rel_path=model["weights_rel_path"],
        )

        req = benchmark_pb2.SingleImagePredictRequest(
            model=proto_model,
            image_data=image_bytes,
            colorize=colorize,
            colormap=colormap,
        )

        try:
            async with grpc.aio.insecure_channel(self.target) as channel:
                stub = benchmark_pb2_grpc.BenchmarkComputeServiceStub(channel)
                resp = await stub.PredictSingleImage(req, timeout=30.0)
                return resp
        except grpc.RpcError as e:
            raise AppException(
                status_code=502,
                message=f"gRPC Compute Error ({e.code().name}): {e.details() or 'Remote error'}",
            )


def get_grpc_client() -> GrpcComputeClient:
    return GrpcComputeClient()
