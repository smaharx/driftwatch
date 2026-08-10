import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


def _utcnow() -> datetime:
    """Return current UTC time. Prefixed with _ — internal use only."""
    return datetime.now(timezone.utc)


class MLModel(Base):
    """
    Registry entry for a monitored ML model.

    Each model has one or more baselines (reference distributions)
    and accumulates runs over time as production data is submitted.
    """

    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    baselines: Mapped[list["Baseline"]] = relationship(
        "Baseline", back_populates="model", cascade="all, delete-orphan"
    )
    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<MLModel id={self.id!r} name={self.name!r}>"


class Baseline(Base):
    """
    Reference distribution for a single feature of a monitored model.

    Stores both summary statistics and the full histogram so drift
    detectors can run without needing the original dataset.
    """

    __tablename__ = "baselines"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # numerical | categorical
    statistics: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # mean, std, min, max, percentiles
    distribution: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # {bins: [...], counts: [...]}
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="baselines")

    def __repr__(self) -> str:
        return f"<Baseline model_id={self.model_id!r} feature={self.feature_name!r}>"


class Run(Base):
    """
    A single drift analysis job against a submitted production batch.

    Status transitions: pending → running → completed | failed
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    overall_drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    drifted_features: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="runs")
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Run id={self.id!r} model_id={self.model_id!r} status={self.status!r}>"


class Alert(Base):
    """
    A drift event that crossed a configured threshold.

    One run can produce multiple alerts — one per drifted feature
    per detector type.
    """

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    detector_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # PSI | KS | CHI2 | JS
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # low | medium | high
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notification_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    run: Mapped["Run"] = relationship("Run", back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert id={self.id!r} feature={self.feature_name!r} severity={self.severity!r}>"
