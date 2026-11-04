import pytest
from src import models
from src.repositories import (
    BenchmarkExperimentRepository,
    DatasetRepository,
    NeuralModelRepository,
    UserRepository,
)


@pytest.mark.asyncio
async def test_user_repository(db_session):
    repo = UserRepository(db_session)
    user = models.User(
        email="repo_user@test.local",
        hashed_password="hashed_secret",
        full_name="Repo User",
        role="RESEARCHER",
    )
    created = await repo.create(user)
    assert created.id is not None
    assert created.email == "repo_user@test.local"

    fetched = await repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.full_name == "Repo User"

    by_email = await repo.get_by_email("repo_user@test.local")
    assert by_email is not None
    assert by_email.id == created.id


@pytest.mark.asyncio
async def test_model_repository(db_session):
    repo = NeuralModelRepository(db_session)
    model = models.NeuralModel(
        name="AdaBinsRepo",
        architecture="transformer",
        backbone="efficientnet",
        weights_rel_path="/weights/adabins.pt",
        is_active=True,
    )
    created = await repo.create(model)
    assert created.id is not None

    all_models = await repo.list_models(is_active=True)
    assert len(all_models) == 1
    assert all_models[0].name == "AdaBinsRepo"

    await repo.delete(created)
    assert await repo.get_by_id(created.id) is None


@pytest.mark.asyncio
async def test_experiment_many_to_many_repository(db_session):
    user_repo = UserRepository(db_session)
    user = await user_repo.create(
        models.User(email="u@test.local", hashed_password="pwd", full_name="U", role="ADMIN")
    )

    dataset_repo = DatasetRepository(db_session)
    dataset = await dataset_repo.create(
        models.Dataset(user_id=user.id, name="kitti", storage_dir="/data/kitti")
    )

    model_repo = NeuralModelRepository(db_session)
    m1 = await model_repo.create(
        models.NeuralModel(
            name="M1", architecture="a1", backbone="b1", weights_rel_path="w1"
        )
    )
    m2 = await model_repo.create(
        models.NeuralModel(
            name="M2", architecture="a2", backbone="b2", weights_rel_path="w2"
        )
    )

    exp_repo = BenchmarkExperimentRepository(db_session)
    exp = models.BenchmarkExperiment(
        user_id=user.id, dataset_id=dataset.id, title="Exp Repo Test"
    )
    created_exp = await exp_repo.create_with_models(exp, [m1.id, m2.id])

    assert created_exp is not None
    assert len(created_exp.model_runs) == 2
    model_names = {run.model.name for run in created_exp.model_runs}
    assert model_names == {"M1", "M2"}

    m1_with_exps = await model_repo.get_model_with_experiments(m1.id)
    assert m1_with_exps is not None
    assert len(m1_with_exps.runs) == 1
    assert m1_with_exps.runs[0].experiment.title == "Exp Repo Test"
