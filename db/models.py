# db/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class MLModel(Base):
    """Registry of every ML model being monitored."""

    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # classification, regression
    feature_names: Mapped[list] = mapped_column(
        JSON, nullable=False
    )  # ["age", "income", "region"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    # Relationships
    baselines: Mapped[list["Baseline"]] = relationship(
        "Baseline", back_populates="model"
    )
    runs: Mapped[list["Run"]] = relationship("Run", back_populates="model")


class Baseline(Base):
    """Reference distribution a model was trained on."""

    __tablename__ = "baselines"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_models.id"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # numerical, categorical
    statistics: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # mean, std, percentiles
    distribution: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # histogram bins + counts
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="baselines")


class Run(Base):
    """One drift analysis job — one batch of production data checked."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_models.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, running, completed, failed
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_results: Mapped[dict] = mapped_column(
        JSON, nullable=True
    )  # full detector output
    overall_drift_score: Mapped[float] = mapped_column(
        Float, nullable=True
    )  # aggregate score
    drifted_features: Mapped[list] = mapped_column(
        JSON, nullable=True
    )  # which features drifted
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    model: Mapped["MLModel"] = relationship("MLModel", back_populates="runs")
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="run")


class Alert(Base):
    """Every drift event that crossed a threshold."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id"), nullable=False
    )
    model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_models.id"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    detector_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # PSI, KS, CHI2, JS
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # low, medium, high
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    notification_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="alerts")
