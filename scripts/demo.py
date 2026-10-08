import asyncio
from sqlalchemy import select
from src.database import AsyncSessionLocal
from src.init_db import drop_db, init_db
from src import crud, models
from scripts.seed import seed

async def run_demo():
    await drop_db()
    await init_db()
    await seed()

    async with AsyncSessionLocal() as db:
        print("=== 1. CRUD: Create ===")
        user = await crud.create_user(db, "operator@test.local", "123", "Operator", "OPERATOR")
        model = await crud.create_model(db, "MiDaS", "dpt", "swin", "/w/midas.pt")
        print(f"Created User: id={user.id}, email={user.email}")
        print(f"Created Model: id={model.id}, name={model.name}")

        print("\n=== 2. CRUD: Read ===")
        fetched_user = await crud.get_user(db, user.id)
        print(f"Fetched User: {fetched_user.full_name} ({fetched_user.role})")

        print("\n=== 3. CRUD: Update ===")
        updated_model = await crud.update_model(db, model.id, is_active=False)
        print(f"Updated Model is_active: {updated_model.is_active}")

        print("\n=== 4. M:N Relation & Query ===")
        res = await db.execute(select(models.ExperimentModelRun))
        runs = res.scalars().all()
        for r in runs:
            print(f"Run id={r.id}: exp_id={r.experiment_id}, model_id={r.model_id}, rmse={r.mean_rmse}")

        print("\n=== 5. CRUD: Delete ===")
        deleted = await crud.delete_model(db, model.id)
        print(f"Model deleted: {deleted}")

if __name__ == "__main__":
    asyncio.run(run_demo())
