from http import HTTPStatus
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.auth import router as auth_router
from src.api.v1.datasets import router as datasets_router
from src.api.v1.experiments import router as experiments_router
from src.api.v1.models import router as models_router
from src.config import settings
from src.core.exceptions import AppException

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="REST API for Deep Learning Neural Depth Benchmark System (Lab 2)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == HTTPStatus.UNAUTHORIZED else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
        headers=headers,
    )


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": settings.APP_VERSION}


api_v1_prefix = "/api/v1"
app.include_router(auth_router, prefix=api_v1_prefix)
app.include_router(models_router, prefix=api_v1_prefix)
app.include_router(datasets_router, prefix=api_v1_prefix)
app.include_router(experiments_router, prefix=api_v1_prefix)
