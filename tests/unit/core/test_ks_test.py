"""Unit tests for KS Test detector."""

import numpy as np
import pytest
from core.detectors.ks_test import KSTestDetector
from core.detectors.base import DriftResult


class TestKSTestDetector:
    """Test KS Test (Kolmogorov-Smirnov) detector."""

    @pytest.fixture
    def detector(self):
        """Create KS Test detector instance."""
        return KSTestDetector(alpha=0.05)

    @pytest.fixture
    def baseline_data(self):
        """Create baseline/reference data."""
        np.random.seed(42)
        return np.random.normal(loc=30, scale=10, size=1000)

    # === Basic Tests ===

    def test_detector_initialization(self, detector):
        """Test KS Test detector initializes correctly."""
        assert detector.name == "KS_TEST"
        assert detector.alpha == 0.05
        assert detector.threshold == 0.05

    def test_set_baseline(self, detector, baseline_data):
        """Test setting baseline/reference data."""
        detector.set_baseline(baseline_data)
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000

    # === No Drift Tests ===

    def test_no_drift_identical_distributions(self, detector, baseline_data):
        """Test KS detects no drift for identical distributions."""
        detector.set_baseline(baseline_data)
        current_data = baseline_data.copy()

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert result.drifted is False
        assert result.p_value > 0.05

    def test_no_drift_very_similar(self, detector, baseline_data):
        """Test KS detects no drift for very similar distributions."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30.1, scale=10.1, size=1000)

        result = detector.detect(current_data)

        assert result.drifted is False
        assert result.p_value > 0.05

    # === Drift Detected Tests ===

    def test_drift_small_shift_detected(self, detector, baseline_data):
        """Test KS detects drift with small shift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=33, scale=10, size=1000)

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert result.p_value < 0.05
        assert result.drifted is True

    def test_drift_major_shift_detected(self, detector, baseline_data):
        """Test KS detects drift with major shift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=50, scale=10, size=1000)

        result = detector.detect(current_data)

        assert result.drifted is True
        assert result.p_value < 0.001
        assert result.score > 0.2

    # === P-Value Tests ===

    def test_p_value_in_range(self, detector, baseline_data):
        """Test p-value is between 0 and 1."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=35, scale=10, size=1000)

        result = detector.detect(current_data)

        assert 0 <= result.p_value <= 1

    def test_ks_statistic_in_range(self, detector, baseline_data):
        """Test KS statistic is between 0 and 1."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert 0 <= result.score <= 1

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

    # === Result Validation ===

    def test_result_has_p_value(self, detector, baseline_data):
        """Test DriftResult includes p_value."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert result.p_value is not None
        assert isinstance(result.p_value, (float, np.floating))

    def test_result_interpretation_contains_pvalue(self, detector, baseline_data):
        """Test interpretation mentions p-value."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=35, scale=10, size=1000)

        result = detector.detect(current_data)

        assert "p" in result.interpretation.lower()

    # === Alpha Threshold Tests ===

    def test_custom_alpha(self):
        """Test custom alpha threshold."""
        detector = KSTestDetector(alpha=0.01)

        assert detector.alpha == 0.01
        assert detector.threshold == 0.01

    def test_stricter_alpha_fewer_alerts(self, baseline_data):
        """Test stricter alpha (0.01) is more conservative."""
        detector_strict = KSTestDetector(alpha=0.01)
        detector_loose = KSTestDetector(alpha=0.1)

        detector_strict.set_baseline(baseline_data)
        detector_loose.set_baseline(baseline_data)

        current_data = np.random.normal(loc=32, scale=10, size=1000)

        result_strict = detector_strict.detect(current_data)
        result_loose = detector_loose.detect(current_data)

        # Loose alpha should alert more often
        assert result_loose.drifted >= result_strict.drifted
