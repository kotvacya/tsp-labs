import base64
from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from src.api.deps import get_current_user, get_model_service, require_role
from src.core.exceptions import BadRequestException
from src.core.grpc_client import get_grpc_client
from src.models import User
from src.schemas.model import (
    NeuralModelCreateRequest,
    NeuralModelResponse,
    NeuralModelUpdateRequest,
)
from src.services.model_service import NeuralModelService

router = APIRouter(prefix="/models", tags=["Neural Models"])


@router.get("", response_model=list[NeuralModelResponse])
async def list_models(
    is_active: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model_service: NeuralModelService = Depends(get_model_service),
) -> list[NeuralModelResponse]:
    models = await model_service.list_models(is_active=is_active, limit=limit, offset=offset)
    return [NeuralModelResponse.model_validate(m) for m in models]


@router.get("/{model_id}", response_model=NeuralModelResponse)
async def get_model(
    model_id: int,
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.get_model(model_id)
    return NeuralModelResponse.model_validate(model)


@router.post("", response_model=NeuralModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    req: NeuralModelCreateRequest,
    current_user: User = Depends(get_current_user),
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.create_model(req)
    return NeuralModelResponse.model_validate(model)


@router.patch("/{model_id}", response_model=NeuralModelResponse)
async def update_model(
    model_id: int,
    req: NeuralModelUpdateRequest,
    current_user: User = Depends(get_current_user),
    model_service: NeuralModelService = Depends(get_model_service),
) -> NeuralModelResponse:
    model = await model_service.update_model(model_id, req)
    return NeuralModelResponse.model_validate(model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    admin_user: User = Depends(require_role("ADMIN")),
    model_service: NeuralModelService = Depends(get_model_service),
) -> None:
    await model_service.delete_model(model_id)


@router.post("/{model_id}/predict")
async def predict_model_depth(
    model_id: int,
    file: UploadFile = File(..., description="Входное RGB изображение (PNG, JPEG, WEBP)"),
    colorize: bool = Query(default=True, description="Визуализация в псевдоцветах"),
    colormap: str = Query(default="magma", description="Палитра: magma, turbo, inferno, plasma, viridis, gray"),
    as_image: bool = Query(default=True, description="Вернуть изображение PNG напрямую или JSON с метаданными и base64"),
    current_user: User = Depends(get_current_user),
    model_service: NeuralModelService = Depends(get_model_service),
):
    """Инференс произвольного изображения на выбранной нейросетевой модели через gRPC."""
    model = await model_service.get_model(model_id)
    if not model.is_active:
        raise BadRequestException(f"Модель '{model.name}' деактивирована")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type and file.content_type.lower() not in allowed_types:
        filename = (file.filename or "").lower()
        if not any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
            raise BadRequestException(
                f"Неподдерживаемый формат файла: {file.content_type}. Разрешены: JPEG, PNG, WEBP, BMP"
            )

    image_bytes = await file.read()
    if not image_bytes:
        raise BadRequestException("Загруженный файл изображения пуст")
    if len(image_bytes) > 25 * 1024 * 1024:
        raise BadRequestException("Размер изображения превышает лимит 25 МБ")

    model_dict = {
        "id": model.id,
        "name": model.name,
        "architecture": model.architecture,
        "backbone": model.backbone,
        "weights_rel_path": model.weights_rel_path,
    }

    client = get_grpc_client()
    resp = await client.predict_single_image(
        model=model_dict,
        image_bytes=image_bytes,
        colorize=colorize,
        colormap=colormap,
    )

    if not resp.success:
        raise BadRequestException(f"Ошибка инференса модели: {resp.error_message}")

    if as_image:
        return Response(
            content=resp.depth_png_data,
            media_type="image/png",
            headers={
                "X-Depth-Min": f"{resp.min_depth:.4f}",
                "X-Depth-Max": f"{resp.max_depth:.4f}",
                "X-Inference-Time-Ms": f"{resp.inference_time_ms:.1f}",
                "X-Image-Width": str(resp.width),
                "X-Image-Height": str(resp.height),
                "Content-Disposition": f'inline; filename="depth_model_{model_id}.png"',
            },
        )

    return {
        "model_id": model_id,
        "model_name": model.name,
        "width": resp.width,
        "height": resp.height,
        "min_depth_meters": resp.min_depth,
        "max_depth_meters": resp.max_depth,
        "inference_time_ms": resp.inference_time_ms,
        "depth_image_base64": base64.b64encode(resp.depth_png_data).decode("utf-8"),
    }
