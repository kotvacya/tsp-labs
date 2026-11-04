from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class NeuralModelCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    architecture: str = Field(min_length=1, max_length=50)
    backbone: str = Field(min_length=1, max_length=50)
    weights_rel_path: str = Field(min_length=1, max_length=255)
    is_active: bool = True


class NeuralModelUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    architecture: str | None = Field(default=None, min_length=1, max_length=50)
    backbone: str | None = Field(default=None, min_length=1, max_length=50)
    weights_rel_path: str | None = Field(default=None, min_length=1, max_length=255)
    is_active: bool | None = None


class NeuralModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    architecture: str
    backbone: str
    weights_rel_path: str
    is_active: bool
    created_at: datetime
