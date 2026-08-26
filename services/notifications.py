
import logging
import httpx
from core.config import settings

logger = logging.getLogger(__name__)


def send_drift_alert(
    model_id: str,
    run_id: str,
    drifted_features: list[str],
    overall_score: float | None,
) -> None:
    """
    Send a Slack webhook notification when drift is detected.

    Silently skips if SLACK_WEBHOOK_URL is not configured.
    Does not raise — notification failure must never crash the analysis.
    """
    if not settings.SLACK_WEBHOOK_URL:
        logger.info("Slack webhook not configured — skipping notification")
        return

    score_text = f"{overall_score:.4f}" if overall_score is not None else "N/A"
    features_text = ", ".join(f"`{f}`" for f in drifted_features)

    message = {
        "text": "🚨 *DriftWatch Alert* — Model drift detected",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 DriftWatch — Drift Detected",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Model ID:*\n`{model_id}`"},
                    {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
                    {"type": "mrkdwn", "text": f"*Overall Score:*\n`{score_text}`"},
                    {"type": "mrkdwn", "text": f"*Drifted Features:*\n{features_text}"},
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Review the drift report and consider retraining.",
                    }
                ],
            },
        ],
    }

    try:
        response = httpx.post(
            settings.SLACK_WEBHOOK_URL,
            json=message,
            timeout=10.0,
        )
        response.raise_for_status()
        logger.info("Slack alert sent for run %s", run_id)
    except httpx.HTTPError as exc:
        logger.error("Failed to send Slack alert: %s", exc)
