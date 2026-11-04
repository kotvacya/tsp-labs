from typing import AsyncGenerator, Callable
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ForbiddenException, UnauthorizedException
from src.core.security import decode_access_token
from src.database import AsyncSessionLocal
from src.models import User
from src.services.auth_service import AuthService
from src.services.dataset_service import DatasetService
from src.services.experiment_service import BenchmarkExperimentService
from src.services.model_service import NeuralModelService

security = HTTPBearer()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(db)


def get_model_service(db: AsyncSession = Depends(get_db)) -> NeuralModelService:
    return NeuralModelService(db)


def get_dataset_service(db: AsyncSession = Depends(get_db)) -> DatasetService:
    return DatasetService(db)


def get_benchmark_service(db: AsyncSession = Depends(get_db)) -> BenchmarkExperimentService:
    return BenchmarkExperimentService(db)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise UnauthorizedException("Could not validate authentication credentials")

    try:
        user_id = int(payload["sub"])
    except (ValueError, TypeError):
        raise UnauthorizedException("Invalid token payload")

    return await auth_service.get_user_by_id(user_id)


def require_role(*roles: str) -> Callable:
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise ForbiddenException(
                f"Operation forbidden: user role '{current_user.role}' lacks required permissions"
            )
        return current_user

    return role_checker
