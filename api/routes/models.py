from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.types import MLModelCreate, MLModelResponse
from db.models import MLModel
from db.session import get_db

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post(
    "",
    response_model=MLModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new ML model for monitoring",
)
def register_model(
    payload: MLModelCreate,
    db: Session = Depends(get_db),
) -> MLModel:
    """
    Register a new ML model in the monitoring registry.

    - **name**: unique model identifier
    - **model_type**: classification | regression | ranking
    - **feature_names**: list of feature names this model uses
    """
    # Check for duplicate name
    existing = db.query(MLModel).filter(MLModel.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model with name '{payload.name}' already exists.",
        )

    model = MLModel(
        name=payload.name,
        description=payload.description,
        model_type=payload.model_type.value,
        feature_names=payload.feature_names,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.get(
    "",
    response_model=list[MLModelResponse],
    summary="List all registered models",
)
def list_models(
    active_only: bool = True,
    db: Session = Depends(get_db),
) -> list[MLModel]:
    """
    Return all registered models.

    - **active_only**: if true, returns only active models (default: true)
    """
    query = db.query(MLModel)
    if active_only:
        query = query.filter(MLModel.is_active == True)
    return query.order_by(MLModel.created_at.desc()).all()


@router.get(
    "/{model_id}",
    response_model=MLModelResponse,
    summary="Get a single model by ID",
)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
) -> MLModel:
    """Return a single model by its UUID."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found.",
        )
    return model


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate a model",
)
def deactivate_model(
    model_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    Soft-delete a model by marking it inactive.

    Does not delete historical runs or alerts.
    """
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found.",
        )
    model.is_active = False
    db.commit()
