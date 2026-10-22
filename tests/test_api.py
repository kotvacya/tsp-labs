import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_and_profile_flow(client: AsyncClient):
    register_payload = {
        "email": "alice@example.com",
        "password": "strongpassword123",
        "full_name": "Alice Researcher",
        "role": "RESEARCHER",
    }
    reg_res = await client.post("/api/v1/auth/register", json=register_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["email"] == "alice@example.com"
    assert "hashed_password" not in user_data
    assert "password" not in user_data

    unauth_res = await client.get("/api/v1/auth/me")
    assert unauth_res.status_code in (401, 403)

    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "strongpassword123"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    assert token

    headers = {"Authorization": f"Bearer {token}"}
    me_res = await client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_models_crud_and_rbac(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "admin@local.host", "password": "adm", "full_name": "Admin", "role": "ADMIN"},
    )
    admin_login = await client.post(
        "/api/v1/auth/login", json={"email": "admin@local.host", "password": "adm"}
    )
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    await client.post(
        "/api/v1/auth/register",
        json={"email": "res@local.host", "password": "res", "full_name": "Res", "role": "RESEARCHER"},
    )
    res_login = await client.post(
        "/api/v1/auth/login", json={"email": "res@local.host", "password": "res"}
    )
    res_token = res_login.json()["access_token"]
    res_headers = {"Authorization": f"Bearer {res_token}"}

    model_payload = {
        "name": "DPT-Large",
        "architecture": "VisionTransformer",
        "backbone": "ViT-L/16",
        "weights_rel_path": "/weights/dpt_large.pt",
        "is_active": True,
    }
    create_res = await client.post("/api/v1/models", json=model_payload, headers=res_headers)
    assert create_res.status_code == 201
    model_id = create_res.json()["id"]

    list_res = await client.get("/api/v1/models")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    patch_res = await client.patch(
        f"/api/v1/models/{model_id}", json={"is_active": False}, headers=res_headers
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["is_active"] is False

    del_res_forbidden = await client.delete(f"/api/v1/models/{model_id}", headers=res_headers)
    assert del_res_forbidden.status_code == 403

    del_res_admin = await client.delete(f"/api/v1/models/{model_id}", headers=admin_headers)
    assert del_res_admin.status_code == 204

    get_res = await client.get(f"/api/v1/models/{model_id}")
    assert get_res.status_code == 404


@pytest.mark.asyncio
async def test_experiment_many_to_many_flow(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "exp_lead@example.com", "password": "lead", "full_name": "Exp Lead", "role": "RESEARCHER"},
    )
    res_login = await client.post(
        "/api/v1/auth/login", json={"email": "exp_lead@example.com", "password": "lead"}
    )
    res_token = res_login.json()["access_token"]
    res_headers = {"Authorization": f"Bearer {res_token}"}

    await client.post(
        "/api/v1/auth/register",
        json={"email": "operator@example.com", "password": "operator123", "full_name": "Data Operator", "role": "OPERATOR"},
    )
    op_login = await client.post(
        "/api/v1/auth/login", json={"email": "operator@example.com", "password": "operator123"}
    )
    op_token = op_login.json()["access_token"]
    op_headers = {"Authorization": f"Bearer {op_token}"}

    forbidden_ds = await client.post(
        "/api/v1/datasets",
        json={"name": "NYU-Depth-V2", "storage_dir": "/datasets/nyu", "total_samples": 1449},
        headers=res_headers,
    )
    assert forbidden_ds.status_code == 403

    ds_res = await client.post(
        "/api/v1/datasets",
        json={"name": "NYU-Depth-V2", "storage_dir": "/datasets/nyu", "total_samples": 1449},
        headers=op_headers,
    )
    assert ds_res.status_code == 201
    dataset_id = ds_res.json()["id"]

    m1_res = await client.post(
        "/api/v1/models",
        json={"name": "AdaBins", "architecture": "trans", "backbone": "eff", "weights_rel_path": "w1"},
        headers=res_headers,
    )
    m2_res = await client.post(
        "/api/v1/models",
        json={"name": "NeWCRFs", "architecture": "crf", "backbone": "swin", "weights_rel_path": "w2"},
        headers=res_headers,
    )
    m1_id = m1_res.json()["id"]
    m2_id = m2_res.json()["id"]

    exp_payload = {
        "title": "Indoor Depth Benchmark",
        "dataset_id": dataset_id,
        "model_ids": [m1_id, m2_id],
    }
    exp_res = await client.post("/api/v1/experiments", json=exp_payload, headers=res_headers)
    assert exp_res.status_code == 201
    exp_data = exp_res.json()
    assert exp_data["title"] == "Indoor Depth Benchmark"
    assert len(exp_data["model_runs"]) == 2

    run_model_names = {run["model"]["name"] for run in exp_data["model_runs"]}
    assert run_model_names == {"AdaBins", "NeWCRFs"}

    exp_id = exp_data["id"]
    fetch_res = await client.get(f"/api/v1/experiments/{exp_id}")
    assert fetch_res.status_code == 200
    assert fetch_res.json()["id"] == exp_id
    assert len(fetch_res.json()["model_runs"]) == 2

    board_res = await client.get(f"/api/v1/experiments/{exp_id}/leaderboard")
    assert board_res.status_code == 200
    leaderboard = board_res.json()
    assert len(leaderboard) == 2
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2
    leaderboard_names = {item["model_name"] for item in leaderboard}
    assert leaderboard_names == {"AdaBins", "NeWCRFs"}
