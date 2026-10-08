import asyncio
from src.database import engine, Base
from src.models import (
    User,
    Dataset,
    DatasetSample,
    NeuralModel,
    BenchmarkExperiment,
    ExperimentModelRun,
    MetricRecord,
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def drop_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

if __name__ == "__main__":
    asyncio.run(init_db())
    print("Database tables created.")
