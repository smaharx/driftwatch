"""Unit tests for PSI detector."""

import numpy as np
import pytest

from core.detectors.base import DriftResult
from core.detectors.psi import PSIDetector


class TestPSIDetector:
    """Test PSI (Population Stability Index) detector."""

    @pytest.fixture
    def detector(self):
        """Create PSI detector instance."""
        return PSIDetector(bins=10, threshold=0.25)

    @pytest.fixture
    def baseline_data(self):
        """Create baseline/reference data."""
        np.random.seed(42)
        return np.random.normal(loc=30, scale=10, size=1000)

    # === Basic Tests ===

    def test_detector_initialization(self, detector):
        """Test PSI detector initializes correctly."""
        assert detector.name == "PSI"
        assert detector.threshold == 0.25
        assert detector.bins == 10

    def test_set_baseline(self, detector, baseline_data):
        """Test setting baseline/reference data."""
        detector.set_baseline(baseline_data)
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000

    def test_baseline_empty_raises_error(self, detector):
        """Test empty baseline raises ValueError."""
        with pytest.raises(ValueError, match="Reference data cannot be empty"):
            detector.set_baseline(np.array([]))

    def test_baseline_none_raises_error(self, detector):
        """Test None baseline raises ValueError."""
        with pytest.raises(ValueError, match="Reference data cannot be empty"):
            detector.set_baseline(None)

    # === No Drift Tests ===

    def test_no_drift_identical_distributions(self, detector, baseline_data):
        """Test PSI detects no drift for identical distributions."""
        detector.set_baseline(baseline_data)
        current_data = baseline_data.copy()

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert result.drifted == False
        assert result.score < 0.05
        assert "No drift" in result.interpretation

    def test_no_drift_very_similar(self, detector, baseline_data):
        """Test PSI detects no drift for very similar distributions."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30.1, scale=10.1, size=1000)

        result = detector.detect(current_data)

        assert result.drifted == False
        assert result.score < 0.15

    # === Small Drift Tests ===

    def test_small_drift_detected(self, detector, baseline_data):
        """Test PSI detects small drift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=33, scale=10, size=1000)

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert 0.05 < result.score < 0.25
        assert (
            "Small drift" in result.interpretation
            or "Moderate drift" in result.interpretation
        )

    # === Significant Drift Tests ===

    def test_significant_drift_detected(self, detector, baseline_data):
        """Test PSI detects significant drift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=50, scale=10, size=1000)

        result = detector.detect(current_data)

        assert result.drifted == True
        assert result.score > detector.threshold
        assert "SIGNIFICANT DRIFT" in result.interpretation

    def test_major_drift_high_psi(self, detector, baseline_data):
        """Test PSI is very high for major drift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=100, scale=5, size=1000)

        result = detector.detect(current_data)

        assert result.drifted == True
        assert result.score > 1.0

    # === Error Handling Tests ===

    def test_detect_without_baseline_raises_error(self, detector):
        """Test detect() raises error if baseline not set."""
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        with pytest.raises(ValueError, match="Baseline not set"):
            detector.detect(current_data)

    def test_detect_empty_current_raises_error(self, detector, baseline_data):
        """Test empty current data raises ValueError."""
        detector.set_baseline(baseline_data)

        with pytest.raises(ValueError, match="Data cannot be empty"):
            detector.detect(np.array([]))

    def test_detect_none_current_raises_error(self, detector, baseline_data):
        """Test None current data raises ValueError."""
        detector.set_baseline(baseline_data)

        with pytest.raises(ValueError, match="Data cannot be None"):
            detector.detect(None)

    def test_detect_invalid_type_raises_error(self, detector, baseline_data):
        """Test invalid data type raises TypeError."""
        detector.set_baseline(baseline_data)

        with pytest.raises(TypeError, match="Expected np.ndarray"):
            detector.detect([1, 2, 3])

    # === Result Validation Tests ===

    def test_result_has_all_fields(self, detector, baseline_data):
        """Test DriftResult has all required fields."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert hasattr(result, "detector_name")
        assert hasattr(result, "score")
        assert hasattr(result, "p_value")
        assert hasattr(result, "drifted")
        assert hasattr(result, "interpretation")

    def test_result_to_dict(self, detector, baseline_data):
        """Test DriftResult serializes to dict."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "detector_name" in result_dict
        assert "score" in result_dict
        assert "drifted" in result_dict

    # === Edge Cases ===

    def test_single_value_baseline(self, detector):
        """Test PSI handles minimal baseline data."""
        baseline = np.array([5.0, 5.0, 5.0])
        detector.set_baseline(baseline)

        # Should not raise error
        result = detector.detect(np.array([5.0, 5.0, 6.0]))
        assert isinstance(result, DriftResult)

    def test_different_sample_sizes(self, detector, baseline_data):
        """Test PSI works with different sample sizes."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=500)  # Half size

        result = detector.detect(current_data)
        assert isinstance(result, DriftResult)

    def test_zero_proportions_handled(self, detector):
        """Test PSI handles bins with zero proportions."""
        baseline = np.array([1, 1, 1, 2, 2, 3])  # Sparse distribution
        detector.set_baseline(baseline)

        current = np.array([1, 1, 2, 2, 3, 4])
        result = detector.detect(current)

        assert not np.isnan(result.score)
        assert not np.isinf(result.score)
