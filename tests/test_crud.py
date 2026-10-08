import pytest
from src import crud

@pytest.mark.asyncio
async def test_user_crud(db_session):
    user = await crud.create_user(db_session, "test@test.local", "secret", "Test User", "RESEARCHER")
    assert user.id is not None
    assert user.email == "test@test.local"

    fetched = await crud.get_user(db_session, user.id)
    assert fetched is not None
    assert fetched.role == "RESEARCHER"

@pytest.mark.asyncio
async def test_model_crud(db_session):
    model = await crud.create_model(db_session, "AdaBinsTest", "trans", "eff", "weights.pt")
    assert model.id is not None
    assert model.is_active is True

    updated = await crud.update_model(db_session, model.id, is_active=False)
    assert updated.is_active is False

    deleted = await crud.delete_model(db_session, model.id)
    assert deleted is True
    assert await crud.get_model(db_session, model.id) is None

@pytest.mark.asyncio
async def test_experiment_many_to_many(db_session):
    user = await crud.create_user(db_session, "exp_user@test.local", "123", "Exp User", "ADMIN")
    dataset = await crud.create_dataset(db_session, user.id, "ds_test", "/path")
    m1 = await crud.create_model(db_session, "M1", "a", "b", "w1")
    m2 = await crud.create_model(db_session, "M2", "a", "b", "w2")

    exp = await crud.create_experiment(db_session, user.id, dataset.id, "Exp 1", [m1.id, m2.id])
    assert exp.id is not None

    exp_with_models = await crud.get_experiment_with_models(db_session, exp.id)
    assert exp_with_models is not None
    assert len(exp_with_models.model_runs) == 2
    model_names = {run.model.name for run in exp_with_models.model_runs}
    assert model_names == {"M1", "M2"}

    m1_with_exps = await crud.get_model_with_experiments(db_session, m1.id)
    assert m1_with_exps is not None
    assert len(m1_with_exps.runs) == 1
    assert m1_with_exps.runs[0].experiment.title == "Exp 1"
