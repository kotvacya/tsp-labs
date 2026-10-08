import asyncio
from src.database import AsyncSessionLocal
from src.init_db import drop_db, init_db
from src import crud
from scripts.seed import seed

async def run_demo():
    await drop_db()
    await init_db()
    await seed()

    async with AsyncSessionLocal() as db:
        print("=== 1. CRUD: Create ===")
        user = await crud.create_user(db, "operator@test.local", "123", "Operator", "OPERATOR")
        m3 = await crud.create_model(db, "MiDaS", "dpt", "swin", "/w/midas.pt")
        print(f"Created User: id={user.id}, email={user.email}")
        print(f"Created Model: id={m3.id}, name={m3.name}")

        print("\n=== 2. CRUD: Read ===")
        fetched_user = await crud.get_user(db, user.id)
        print(f"Fetched User: {fetched_user.full_name} ({fetched_user.role})")

        print("\n=== 3. CRUD: Update ===")
        updated_model = await crud.update_model(db, m3.id, is_active=False)
        print(f"Updated Model is_active: {updated_model.is_active}")

        print("\n=== 4. Many-to-Many (M:N, 2x3 matrix) ===")
        # Эксперимент 1 (из seed) содержит модели 1 и 2
        # Создаем Эксперимент 2, содержащий 3 модели: 1, 2 и 3
        exp2 = await crud.create_experiment(
            db,
            user_id=user.id,
            dataset_id=1,
            title="Night Benchmark",
            model_ids=[1, 2, m3.id],
        )

        exp1_obj = await crud.get_experiment_with_models(db, experiment_id=1)
        print(f"Experiment id={exp1_obj.id} ('{exp1_obj.title}') models ({len(exp1_obj.model_runs)}):")
        for run in exp1_obj.model_runs:
            print(f"  -> model: {run.model.name}")

        exp2_obj = await crud.get_experiment_with_models(db, experiment_id=exp2.id)
        print(f"Experiment id={exp2_obj.id} ('{exp2_obj.title}') models ({len(exp2_obj.model_runs)}):")
        for run in exp2_obj.model_runs:
            print(f"  -> model: {run.model.name}")

        m1_obj = await crud.get_model_with_experiments(db, model_id=1)
        print(f"Model id={m1_obj.id} ('{m1_obj.name}') experiments ({len(m1_obj.runs)}):")
        for run in m1_obj.runs:
            print(f"  -> exp id={run.experiment.id}: '{run.experiment.title}'")

        m3_obj = await crud.get_model_with_experiments(db, model_id=3)
        print(f"Model id={m3_obj.id} ('{m3_obj.name}') experiments ({len(m3_obj.runs)}):")
        for run in m3_obj.runs:
            print(f"  -> exp id={run.experiment.id}: '{run.experiment.title}'")

        print("\n=== 5. CRUD: Delete ===")
        deleted = await crud.delete_model(db, m3.id)
        print(f"Model deleted: {deleted}")

if __name__ == "__main__":
    asyncio.run(run_demo())
