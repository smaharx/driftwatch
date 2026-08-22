import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.types import AlertAcknowledge, AlertResponse
from db.models import Alert, MLModel
from db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get(
    "",
    response_model=list[AlertResponse],
    summary="List all drift alerts",
)
def list_alerts(
    model_id: str | None = None,
    severity: str | None = None,
    acknowledged: bool | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[Alert]:
    """
    Return drift alerts with optional filters.

    - **model_id**: filter by model
    - **severity**: filter by low | medium | high
    - **acknowledged**: filter by acknowledgement status
    - **limit**: max results (default 50)
    """
    query = db.query(Alert)

    if model_id:
        query = query.filter(Alert.model_id == model_id)
    if severity:
        query = query.filter(Alert.severity == severity)
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)

    return query.order_by(Alert.created_at.desc()).limit(limit).all()


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get a single alert by ID",
)
def get_alert(
    alert_id: str,
    db: Session = Depends(get_db),
) -> Alert:
    """Return a single alert by its UUID."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return alert


@router.patch(
    "/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge a drift alert",
)
def acknowledge_alert(
    alert_id: str,
    payload: AlertAcknowledge,
    db: Session = Depends(get_db),
) -> Alert:
    """
    Mark an alert as acknowledged.

    Acknowledged alerts are excluded from default dashboard counts
    but remain in the audit log.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )

    alert.acknowledged = payload.acknowledged
    db.commit()
    db.refresh(alert)

    logger.info("Alert %s acknowledged=%s", alert_id, payload.acknowledged)
    return alert


@router.get(
    "/model/{model_id}/summary",
    response_model=dict,
    summary="Get alert summary for a model",
)
def alert_summary(
    model_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Return alert counts grouped by severity for a model.

    Used by the dashboard overview cards.
    """
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found.",
        )

    alerts = db.query(Alert).filter(Alert.model_id == model_id).all()

    return {
        "model_id": model_id,
        "model_name": model.name,
        "total": len(alerts),
        "unacknowledged": sum(1 for a in alerts if not a.acknowledged),
        "by_severity": {
            "high": sum(1 for a in alerts if a.severity == "high"),
            "medium": sum(1 for a in alerts if a.severity == "medium"),
            "low": sum(1 for a in alerts if a.severity == "low"),
        },
    }
