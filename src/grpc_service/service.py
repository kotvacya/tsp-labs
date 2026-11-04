"""Реализация сервиса gRPC BenchmarkComputeService."""

import sys
import time
from pathlib import Path

from src.grpc_gen import benchmark_pb2, benchmark_pb2_grpc
from src.grpc_service.runner import ModelBenchmarkRunner, detect_device


class BenchmarkComputeServicer(benchmark_pb2_grpc.BenchmarkComputeServiceServicer):
    def __init__(self, root_dir: Path | None = None):
        self.runner = ModelBenchmarkRunner(root_dir=root_dir)

    def HealthCheck(self, request, context):
        device = detect_device()
        print(f"[gRPC Compute] HealthCheck -> status: SERVING, device: {device}", flush=True)
        return benchmark_pb2.HealthResponse(
            status="SERVING",
            device=device,
            version="1.0.0",
        )

    def ExecuteBenchmark(self, request: benchmark_pb2.BenchmarkRequest, context):
        t_start = time.perf_counter()
        device = detect_device()
        print("\n" + "=" * 70, flush=True)
        print("[gRPC Compute] ExecuteBenchmark request received", flush=True)
        print(f"  Experiment ID: {request.experiment_id}", flush=True)
        print(f"  Dataset: {request.dataset_name} (id={request.dataset_id})", flush=True)
        print(f"  Models: {len(request.models)}", flush=True)
        print(f"  Samples: {len(request.samples)} (max_samples={request.max_samples})", flush=True)
        print(f"  Device: {device}", flush=True)
        print("=" * 70, flush=True)

        samples_list = [
            (s.sample_index, s.rgb_rel_path, s.gt_depth_rel_path)
            for s in request.samples
        ]

        if request.max_samples > 0:
            samples_list = samples_list[: request.max_samples]

        model_results = []

        for m in request.models:
            print(f"[gRPC Compute] Evaluating model '{m.model_name}' [{m.architecture}/{m.backbone}] on {len(samples_list)} samples...", flush=True)
            res = self.runner.run_model_on_samples(
                model_id=m.model_id,
                model_name=m.model_name,
                architecture=m.architecture,
                backbone=m.backbone,
                weights_rel_path=m.weights_rel_path,
                dataset_storage_dir=request.dataset_storage_dir,
                samples=samples_list,
            )

            if res["run_status"] == "FAILED":
                print(f"[gRPC Compute] Model '{m.model_name}' FAILED: {res['error_message']}", flush=True)
            else:
                print(
                    f"[gRPC Compute] Completed '{m.model_name}': RMSE={res['mean_rmse']:.4f}, "
                    f"SILog={res['mean_silog']:.2f}, AbsRel={res['mean_absrel']:.4f}, "
                    f"Delta1={res['mean_delta_1']:.4f}, AvgLatency={res['avg_inference_time_ms']:.1f}ms",
                    flush=True,
                )

            sample_proto_metrics = [
                benchmark_pb2.SampleMetricResult(
                    sample_index=sm["sample_index"],
                    rmse=sm["rmse"],
                    silog=sm["silog"],
                    abs_rel=sm["abs_rel"],
                    delta_1=sm["delta_1"],
                    inference_time_ms=sm["inference_time_ms"],
                )
                for sm in res["sample_metrics"]
            ]

            model_results.append(
                benchmark_pb2.ModelExecutionResult(
                    model_id=res["model_id"],
                    model_name=res["model_name"],
                    run_status=res["run_status"],
                    error_message=res["error_message"],
                    mean_rmse=res["mean_rmse"],
                    mean_silog=res["mean_silog"],
                    mean_absrel=res["mean_absrel"],
                    mean_delta_1=res["mean_delta_1"],
                    avg_inference_time_ms=res["avg_inference_time_ms"],
                    sample_metrics=sample_proto_metrics,
                )
            )

        duration = time.perf_counter() - t_start
        print(f"[gRPC Compute] Benchmark finished in {duration:.2f}s\n", flush=True)

        return benchmark_pb2.BenchmarkResponse(
            experiment_id=request.experiment_id,
            status="FINISHED",
            total_duration_seconds=duration,
            results=model_results,
        )

    def PredictSingleImage(self, request: benchmark_pb2.SingleImagePredictRequest, context):
        m = request.model
        print(
            f"[gRPC Compute] PredictSingleImage request: model '{m.model_name}' [{m.architecture}/{m.backbone}], "
            f"image size={len(request.image_data)} bytes, colormap={request.colormap or 'magma'}",
            flush=True,
        )

        res = self.runner.predict_single_image(
            architecture=m.architecture,
            backbone=m.backbone,
            weights_rel_path=m.weights_rel_path,
            image_bytes=request.image_data,
            colorize=request.colorize,
            colormap=request.colormap,
        )

        if not res["success"]:
            print(f"[gRPC Compute] PredictSingleImage FAILED: {res['error_message']}", flush=True)
        else:
            print(
                f"[gRPC Compute] PredictSingleImage COMPLETED: {res['width']}x{res['height']}, "
                f"depth [{res['min_depth']:.2f}m - {res['max_depth']:.2f}m] in {res['inference_time_ms']:.1f}ms",
                flush=True,
            )

        return benchmark_pb2.SingleImagePredictResponse(
            success=res["success"],
            error_message=res["error_message"],
            depth_png_data=res["depth_png_data"],
            width=res["width"],
            height=res["height"],
            min_depth=res["min_depth"],
            max_depth=res["max_depth"],
            inference_time_ms=res["inference_time_ms"],
        )
