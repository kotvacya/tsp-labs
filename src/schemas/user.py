from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=3)
    full_name: str = Field(min_length=1, max_length=150)
    role: str = Field(default="RESEARCHER", max_length=30)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
