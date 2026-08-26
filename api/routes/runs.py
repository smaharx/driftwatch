import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.types import BaselineCreate, BaselineResponse, RunCreate, RunResponse
from db.models import Baseline, MLModel, Run
from db.session import get_db
from services.drift_analyzer import DriftAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["runs"])


@router.post(
    "/models/{model_id}/baseline",
    response_model=list[BaselineResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload baseline data for a model",
)
def upload_baseline(
    model_id: str,
    payload: BaselineCreate,
    db: Session = Depends(get_db),
) -> list[Baseline]:
    """
    Upload reference (training) data distributions for a registered model.

    This must be called before submitting any drift analysis runs.
    Re-uploading replaces the existing baseline for each feature.
    """
    model = (
        db.query(MLModel)
        .filter(
            MLModel.id == model_id,
            MLModel.is_active == True,
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found or is inactive.",
        )

    submitted_names = {f.feature_name for f in payload.features}
    registered_names = set(model.feature_names)
    unknown = submitted_names - registered_names

    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown features not registered with this model: {sorted(unknown)}",
        )

    analyzer = DriftAnalyzer(db)
    baselines = analyzer.store_baseline(
        model=model,
        features=[f.model_dump() for f in payload.features],
    )
    return baselines




@router.post(
    "/models/{model_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a production data batch for drift analysis",
)
def create_run(
    model_id: str,
    payload: RunCreate,
    db: Session = Depends(get_db),
) -> Run:
    """
    Submit a batch of production data for drift analysis.

    Runs analysis synchronously and returns the completed report.
    Architecture supports async via Celery — see workers/tasks.py.
    """
    model = (
        db.query(MLModel)
        .filter(
            MLModel.id == model_id,
            MLModel.is_active == True,  # noqa: E712
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found or is inactive.",
        )

    has_baseline = db.query(Baseline).filter(Baseline.model_id == model_id).first()

    if not has_baseline:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No baseline found for this model. Upload baseline data first.",
        )

    sample_size = len(payload.features[0].values) if payload.features else 0

    run = Run(
        model_id=model_id,
        status="pending",
        sample_size=sample_size,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    analyzer = DriftAnalyzer(db)
    completed_run = analyzer.analyze(
        run=run,
        features=[f.model_dump() for f in payload.features],
    )

    return completed_run 

@router.get(
    "/models/{model_id}/runs",
    response_model=list[RunResponse],
    summary="List all runs for a model",
)
def list_runs(
    model_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[Run]:
    """Return the most recent runs for a model, newest first."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found.",
        )

    return (
        db.query(Run)
        .filter(Run.model_id == model_id)
        .order_by(Run.created_at.desc())
        .limit(limit)
        .all()
    )


@router.get(
    "/runs/{run_id}",
    response_model=RunResponse,
    summary="Get a single drift analysis run by ID",
)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> Run:
    """Return the full drift report for a single run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return run
