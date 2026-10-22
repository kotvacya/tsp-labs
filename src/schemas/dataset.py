from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    storage_dir: str = Field(min_length=1, max_length=255)
    total_samples: int = Field(default=0, ge=0)


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    name: str
    description: str | None
    storage_dir: str
    total_samples: int
    created_at: datetime
