from fastapi import APIRouter, Depends, status
from src.api.deps import get_auth_service, get_current_user
from src.models import User
from src.schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from src.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    req: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    user = await auth_service.register(req)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    req: UserLoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await auth_service.authenticate(req)


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)
