
import logging
from celery import shared_task
from sqlalchemy.orm import Session

from db.models import Run
from db.session import SessionLocal
from services.drift_analyzer import DriftAnalyzer
from services.notifications import send_drift_alert

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="workers.tasks.analyze_drift",
)
def analyze_drift(self, run_id: str, features: list[dict]) -> dict:
    """
    Async Celery task — runs drift analysis for a submitted batch.

    Retries up to 3 times on failure with 60s delay.
    Sends Slack notification if drift is detected.
    """
    db: Session = SessionLocal()

    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            logger.error("Run %s not found", run_id)
            return {"error": "run not found"}

        analyzer = DriftAnalyzer(db)
        completed_run = analyzer.analyze(run=run, features=features)

        if completed_run.drifted_features:
            send_drift_alert(
                model_id=completed_run.model_id,
                run_id=run_id,
                drifted_features=completed_run.drifted_features,
                overall_score=completed_run.overall_drift_score,
            )

        return {
            "run_id": run_id,
            "status": completed_run.status,
            "drifted_features": completed_run.drifted_features,
        }

    except Exception as exc:
        logger.exception("Task failed for run %s", run_id)
        raise self.retry(exc=exc)

    finally:
        db.close()
