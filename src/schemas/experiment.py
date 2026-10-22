from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from src.schemas.model import NeuralModelResponse


class ExperimentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    dataset_id: int
    model_ids: list[int] = Field(min_length=1)


class ExperimentModelRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    model_id: int
    model: NeuralModelResponse
    mean_silog: float | None = None
    mean_absrel: float | None = None
    mean_rmse: float | None = None
    mean_delta_1: float | None = None
    avg_inference_time_ms: float | None = None
    run_status: str


class ExperimentListItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    dataset_id: int
    title: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    dataset_id: int
    title: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    model_runs: list[ExperimentModelRunResponse] = []


class LeaderboardItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    model_id: int
    model_name: str
    architecture: str
    backbone: str
    mean_rmse: float | None = None
    mean_silog: float | None = None
    mean_absrel: float | None = None
    mean_delta_1: float | None = None
    avg_inference_time_ms: float | None = None
    run_status: str
