from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ─── Enums ────────────────────────────────────────────────────────────────────


class ModelType(str, Enum):
    classification = "classification"
    regression = "regression"
    ranking = "ranking"


class FeatureType(str, Enum):
    numerical = "numerical"
    categorical = "categorical"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DetectorType(str, Enum):
    PSI = "PSI"
    KS = "KS"
    CHI2 = "CHI2"
    JS = "JS"


# ─── ML Model schemas ─────────────────────────────────────────────────────────


class MLModelCreate(BaseModel):
    """Request body for registering a new model."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    model_type: ModelType
    feature_names: list[str] = Field(..., min_length=1)

    @field_validator("feature_names")
    @classmethod
    def feature_names_must_be_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("feature_names must be unique")
        return v


class MLModelResponse(BaseModel):
    """Response body for a registered model."""

    id: str
    name: str
    description: str | None
    model_type: str
    feature_names: list[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ─── Baseline schemas ─────────────────────────────────────────────────────────


class BaselineFeatureData(BaseModel):
    """Reference data for a single feature."""

    feature_name: str = Field(..., min_length=1)
    feature_type: FeatureType
    values: list[float | str | int] = Field(..., min_length=10)


class BaselineCreate(BaseModel):
    """Request body for uploading baseline data for a model."""

    features: list[BaselineFeatureData] = Field(..., min_length=1)


class BaselineResponse(BaseModel):
    """Response confirming baseline was stored."""

    id: str
    model_id: str
    feature_name: str
    feature_type: str
    sample_size: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Run schemas ──────────────────────────────────────────────────────────────


class RunCreate(BaseModel):
    """Request body for submitting a production data batch for drift analysis."""

    features: list[BaselineFeatureData] = Field(..., min_length=1)


class DriftFeatureResult(BaseModel):
    """Drift result for a single feature."""

    feature_name: str
    feature_type: str
    drifted: bool
    detector: str
    score: float
    threshold: float
    p_value: float | None
    interpretation: str


class RunResponse(BaseModel):
    """Response for a completed drift analysis run."""

    id: str
    model_id: str
    status: str
    sample_size: int
    overall_drift_score: float | None
    drifted_features: list[str] | None
    drift_results: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


# ─── Alert schemas ────────────────────────────────────────────────────────────


class AlertResponse(BaseModel):
    """Response for a drift alert."""

    id: str
    run_id: str
    model_id: str
    feature_name: str
    detector_type: str
    drift_score: float
    threshold: float
    severity: str
    acknowledged: bool
    notification_sent: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    """Request body to acknowledge an alert."""

    acknowledged: bool = True


# ─── Generic response schemas ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list endpoints."""

    items: list[Any]
    total: int
    page: int
    page_size: int
