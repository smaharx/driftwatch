"""
Base class for all drift detectors.
All detectors must implement this interface.
"""

from abc import ABC, abstractmethod

import numpy as np


class DriftDetector(ABC):
    """
    Abstract base class for drift detection methods.

    All drift detectors must:
    1. Inherit from this class
    2. Implement detect() method
    3. Return standardized DriftResult
    """

    def __init__(self, name: str, threshold: float = 0.05):
        """
        Initialize drift detector.

        Args:
            name: Detector name (e.g., "PSI", "KS_TEST")
            threshold: Alert threshold (varies by detector)
        """
        self.name = name
        self.threshold = threshold
        self.reference_data = None

    def set_baseline(self, reference_data: np.ndarray) -> None:
        """
        Set the baseline/reference distribution.

        Args:
            reference_data: Training data distribution
        """
        if reference_data is None or len(reference_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self.reference_data = reference_data

    @abstractmethod
    def detect(self, current_data: np.ndarray) -> "DriftResult":
        """
        Detect drift in current data vs reference.

        Args:
            current_data: New data to check for drift

        Returns:
            DriftResult object with drift scores and interpretation
        """

    def _validate_data(self, data: np.ndarray) -> bool:
        """
        Validate input data.

        Args:
            data: Data to validate

        Returns:
            True if valid, raises exception if not
        """
        if data is None:
            raise ValueError("Data cannot be None")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(data)}")

        return True


class DriftResult:
    """
    Standardized result from any drift detector.

    Attributes:
        detector_name: Name of detector used
        score: Primary drift score (varies by detector)
        p_value: Statistical p-value (if applicable)
        threshold: Threshold for alerting
        drifted: Boolean, True if drift detected
        interpretation: Human-readable explanation
    """

    def __init__(
        self,
        detector_name: str,
        score: float,
        p_value: float = None,
        threshold: float = None,
        drifted: bool = False,
        interpretation: str = "",
    ):
        self.detector_name = detector_name
        self.score = score
        self.p_value = p_value
        self.threshold = threshold
        self.drifted = drifted
        self.interpretation = interpretation

    def __repr__(self):
        return (
            f"DriftResult("
            f"detector={self.detector_name}, "
            f"score={self.score:.4f}, "
            f"drifted={self.drifted})"
        )

    def to_dict(self) -> dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "detector_name": self.detector_name,
            "score": float(self.score),
            "p_value": float(self.p_value) if self.p_value else None,
            "threshold": float(self.threshold) if self.threshold else None,
            "drifted": bool(self.drifted),
            "interpretation": self.interpretation,
        }
