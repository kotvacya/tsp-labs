from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tspy_db"
    TEST_DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tspy_test_db"

    SECRET_KEY: str = "super-secret-key-change-in-production-benchmark-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    APP_TITLE: str = "MDE-Bench Platform REST API"
    APP_VERSION: str = "0.2.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
