from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import AlreadyExistsException, NotFoundException, UnauthorizedException
from src.core.security import create_access_token, hash_password, verify_password
from src.models import User
from src.repositories.user_repo import UserRepository
from src.schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)

    async def register(self, req: UserRegisterRequest) -> User:
        existing = await self.user_repo.get_by_email(req.email)
        if existing:
            raise AlreadyExistsException(f"User with email '{req.email}' already exists")

        user = User(
            email=req.email,
            hashed_password=hash_password(req.password),
            full_name=req.full_name,
            role=req.role,
        )
        return await self.user_repo.create(user)

    async def authenticate(self, req: UserLoginRequest) -> TokenResponse:
        user = await self.user_repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.hashed_password):
            raise UnauthorizedException("Invalid email or password")

        token = create_access_token(
            {"sub": str(user.id), "email": user.email, "role": user.role}
        )
        return TokenResponse(access_token=token)

    async def get_user_by_id(self, user_id: int) -> User:
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException(f"User with id={user_id} not found")
        return user
