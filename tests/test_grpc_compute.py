import asyncio
from concurrent import futures
from pathlib import Path
import grpc
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.grpc_client import GrpcComputeClient
from src.core.seed import seed_initial_data
from src.grpc_gen import benchmark_pb2, benchmark_pb2_grpc
from src.grpc_service.service import BenchmarkComputeServicer
from src.models import Dataset, NeuralModel


@pytest.fixture(scope="module")
def grpc_server():
    """Запуск тестового gRPC сервера на порту 50055."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = BenchmarkComputeServicer(root_dir=Path.cwd())
    benchmark_pb2_grpc.add_BenchmarkComputeServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:50055")
    server.start()
    yield f"127.0.0.1:{port}"
    server.stop(grace=0)


@pytest.mark.asyncio
async def test_grpc_direct_rpc_calls(grpc_server):
    async with grpc.aio.insecure_channel(grpc_server) as channel:
        stub = benchmark_pb2_grpc.BenchmarkComputeServiceStub(channel)

        health_resp = await stub.HealthCheck(benchmark_pb2.HealthRequest())
        assert health_resp.status == "SERVING"
        assert health_resp.device

        req = benchmark_pb2.BenchmarkRequest(
            experiment_id=101,
            dataset_id=1,
            dataset_name="NYU-Test",
            dataset_storage_dir="data/nyu_depth_v2",
            models=[
                benchmark_pb2.ModelExecutionConfig(
                    model_id=1,
                    model_name="IEBins-Tiny-NYU",
                    architecture="IEBins",
                    backbone="Swin-Tiny",
                    weights_rel_path="checkpoints/iebins/nyu_T.pth",
                ),
                benchmark_pb2.ModelExecutionConfig(
                    model_id=2,
                    model_name="IEBins-Large-NYU",
                    architecture="IEBins",
                    backbone="Swin-Large",
                    weights_rel_path="checkpoints/iebins/nyu_L.pth",
                ),
            ],
            samples=[
                benchmark_pb2.SamplePayload(
                    sample_index=0,
                    rgb_rel_path="extracted/rgb/sample_0000.png",
                    gt_depth_rel_path="extracted/depth/sample_0000.png",
                ),
                benchmark_pb2.SamplePayload(
                    sample_index=1,
                    rgb_rel_path="extracted/rgb/sample_0001.png",
                    gt_depth_rel_path="extracted/depth/sample_0001.png",
                ),
            ],
            max_samples=2,
        )

        resp = await stub.ExecuteBenchmark(req)
        assert resp.experiment_id == 101
        assert resp.status == "FINISHED"
        assert len(resp.results) == 2

        res_tiny = resp.results[0]
        assert res_tiny.model_id == 1
        assert res_tiny.run_status == "COMPLETED"
        assert res_tiny.mean_rmse > 0
        assert res_tiny.avg_inference_time_ms > 0
        assert len(res_tiny.sample_metrics) == 2

        # Прямой вызов PredictSingleImage
        img_path = Path("data/nyu_depth_v2/extracted/rgb/sample_0000.png")
        if img_path.exists():
            img_bytes = img_path.read_bytes()
            pred_req = benchmark_pb2.SingleImagePredictRequest(
                model=benchmark_pb2.ModelExecutionConfig(
                    model_id=1,
                    model_name="IEBins-Tiny-NYU",
                    architecture="IEBins",
                    backbone="Swin-Tiny",
                    weights_rel_path="checkpoints/iebins/nyu_T.pth",
                ),
                image_data=img_bytes,
                colorize=True,
                colormap="magma",
            )
            pred_resp = await stub.PredictSingleImage(pred_req)
            assert pred_resp.success
            assert len(pred_resp.depth_png_data) > 0
            assert pred_resp.width == 640
            assert pred_resp.height == 480
            assert pred_resp.min_depth > 0
            assert pred_resp.max_depth > pred_resp.min_depth


@pytest.mark.asyncio
async def test_full_experiment_run_and_leaderboard_flow(client: AsyncClient, db_session: AsyncSession, grpc_server, monkeypatch):
    monkeypatch.setattr("src.config.settings.GRPC_COMPUTE_HOST", "127.0.0.1")
    monkeypatch.setattr("src.config.settings.GRPC_COMPUTE_PORT", 50055)

    await seed_initial_data(db_session)

    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "researcher@mdebench.com", "password": "researcherpass"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    status_res = await client.get("/api/v1/experiments/compute/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "SERVING"

    datasets_res = await client.get("/api/v1/datasets")
    assert datasets_res.status_code == 200
    datasets = datasets_res.json()
    assert len(datasets) >= 2
    nyu_dataset = next(d for d in datasets if "NYU" in d["name"])

    models_res = await client.get("/api/v1/models")
    assert models_res.status_code == 200
    models = models_res.json()
    assert len(models) >= 2
    m1_id = models[0]["id"]
    m2_id = models[1]["id"]

    exp_payload = {
        "title": "IEBins Tiny vs Large Benchmark Test",
        "dataset_id": nyu_dataset["id"],
        "model_ids": [m1_id, m2_id],
    }
    create_res = await client.post("/api/v1/experiments", json=exp_payload, headers=headers)
    assert create_res.status_code == 201
    exp_data = create_res.json()
    exp_id = exp_data["id"]
    assert exp_data["status"] == "PENDING"
    assert exp_data["model_runs"][0]["mean_rmse"] is None

    run_res = await client.post(
        f"/api/v1/experiments/{exp_id}/run?max_samples=3",
        headers=headers,
    )
    assert run_res.status_code == 200
    finished_exp = run_res.json()
    assert finished_exp["status"] == "FINISHED"
    assert finished_exp["finished_at"] is not None

    for run in finished_exp["model_runs"]:
        assert run["run_status"] == "COMPLETED"
        assert run["mean_rmse"] is not None
        assert run["mean_rmse"] > 0
        assert run["mean_delta_1"] is not None
        assert run["avg_inference_time_ms"] > 0

    lb_res = await client.get(f"/api/v1/experiments/{exp_id}/leaderboard?sort_by=rmse&order=asc")
    assert lb_res.status_code == 200
    leaderboard = lb_res.json()
    assert len(leaderboard) == 2
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2
    assert leaderboard[0]["mean_rmse"] <= leaderboard[1]["mean_rmse"]

    # Проверка отсутствия дублирования метрик при повторном запуске эксперимента
    from sqlalchemy import func, select
    from src.models import MetricRecord

    count_after_first_run = await db_session.scalar(select(func.count()).select_from(MetricRecord))
    assert count_after_first_run > 0

    rerun_res = await client.post(
        f"/api/v1/experiments/{exp_id}/run?max_samples=3",
        headers=headers,
    )
    assert rerun_res.status_code == 200
    rerun_exp = rerun_res.json()
    assert rerun_exp["status"] == "FINISHED"

    count_after_rerun = await db_session.scalar(select(func.count()).select_from(MetricRecord))
    assert count_after_rerun == count_after_first_run


@pytest.mark.asyncio
async def test_predict_single_image_api_endpoint(client: AsyncClient, db_session: AsyncSession, grpc_server, monkeypatch):
    monkeypatch.setattr("src.config.settings.GRPC_COMPUTE_HOST", "127.0.0.1")
    monkeypatch.setattr("src.config.settings.GRPC_COMPUTE_PORT", 50055)

    await seed_initial_data(db_session)

    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "researcher@mdebench.com", "password": "researcherpass"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    models_res = await client.get("/api/v1/models")
    m_id = models_res.json()[0]["id"]

    import io
    from PIL import Image

    # Тестовая картинка нестандартного размера (220x170)
    dummy = Image.new("RGB", (220, 170), color=(100, 150, 200))
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    png_data = buf.getvalue()

    # 1. Запрос с as_image=true (бинарный PNG с заголовками)
    files = {"file": ("test.png", png_data, "image/png")}
    res_img = await client.post(
        f"/api/v1/models/{m_id}/predict?colorize=true&colormap=turbo&as_image=true",
        headers=headers,
        files=files,
    )
    assert res_img.status_code == 200
    assert res_img.headers["content-type"] == "image/png"
    assert "X-Depth-Min" in res_img.headers
    assert "X-Depth-Max" in res_img.headers
    assert res_img.headers["X-Image-Width"] == "220"
    assert res_img.headers["X-Image-Height"] == "170"
    assert len(res_img.content) > 0

    # 2. Запрос с as_image=false (JSON с base64)
    files_json = {"file": ("test.png", png_data, "image/png")}
    res_json = await client.post(
        f"/api/v1/models/{m_id}/predict?colorize=true&colormap=magma&as_image=false",
        headers=headers,
        files=files_json,
    )
    assert res_json.status_code == 200
    data = res_json.json()
    assert data["model_id"] == m_id
    assert data["width"] == 220
    assert data["height"] == 170
    assert data["min_depth_meters"] > 0
    assert "depth_image_base64" in data

    # 3. Валидация: пустой файл
    empty_files = {"file": ("empty.png", b"", "image/png")}
    res_empty = await client.post(
        f"/api/v1/models/{m_id}/predict",
        headers=headers,
        files=empty_files,
    )
    assert res_empty.status_code == 400
