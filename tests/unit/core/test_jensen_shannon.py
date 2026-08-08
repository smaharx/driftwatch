"""Unit tests for Jensen-Shannon divergence detector."""

import numpy as np
import pytest

from core.detectors.base import DriftResult
from core.detectors.jensen_shannon import JensenShannonDetector


class TestJensenShannonDetector:
    """Test Jensen-Shannon divergence detector."""

    @pytest.fixture
    def detector(self):
        """Create Jensen-Shannon detector instance."""
        return JensenShannonDetector(bins=10, threshold=0.15)

    @pytest.fixture
    def baseline_data(self):
        """Create baseline/reference data."""
        np.random.seed(42)
        return np.random.normal(loc=30, scale=10, size=1000)

    # === Basic Tests ===

    def test_detector_initialization(self, detector):
        """Test Jensen-Shannon detector initializes correctly."""
        assert detector.name == "JENSEN_SHANNON"
        assert detector.threshold == 0.15
        assert detector.bins == 10

    def test_set_baseline(self, detector, baseline_data):
        """Test setting baseline/reference data."""
        detector.set_baseline(baseline_data)
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000

    # === No Drift Tests ===

    def test_no_drift_identical_distributions(self, detector, baseline_data):
        """Test JS detects no drift for identical distributions."""
        detector.set_baseline(baseline_data)
        current_data = baseline_data.copy()

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert result.drifted is False
        assert result.score < 0.05

    def test_no_drift_very_similar(self, detector, baseline_data):
        """Test JS detects no drift for very similar distributions."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30.1, scale=10.1, size=1000)

        result = detector.detect(current_data)

        assert result.drifted is False
        assert result.score < 0.15

    # === Drift Detected Tests ===

    def test_drift_small_shift(self, detector, baseline_data):
        """Test JS detects drift with small shift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=33, scale=10, size=1000)

        result = detector.detect(current_data)

        assert isinstance(result, DriftResult)
        assert 0.05 < result.score < 0.25

    def test_drift_major_shift(self, detector, baseline_data):
        """Test JS detects drift with major shift."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=50, scale=10, size=1000)

        result = detector.detect(current_data)

        assert result.drifted is True
        assert result.score > detector.threshold

    # === Range Tests ===

    def test_js_score_in_range(self, detector, baseline_data):
        """Test Jensen-Shannon score is between 0 and 1."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert 0 <= result.score <= 1

    def test_js_never_negative(self, detector, baseline_data):
        """Test JS score is never negative."""
        detector.set_baseline(baseline_data)

        for _ in range(10):
            current_data = np.random.normal(loc=30, scale=10, size=1000)
            result = detector.detect(current_data)
            assert result.score >= 0

    # === Symmetry Test ===

    def test_symmetry_js_divergence(self):
        """Test Jensen-Shannon symmetry: JSD(P||Q) = JSD(Q||P)."""
        detector1 = JensenShannonDetector()
        detector2 = JensenShannonDetector()

        np.random.seed(42)
        data1 = np.random.normal(loc=30, scale=10, size=1000)
        data2 = np.random.normal(loc=35, scale=10, size=1000)

        detector1.set_baseline(data1)
        result1 = detector1.detect(data2)

        detector2.set_baseline(data2)
        result2 = detector2.detect(data1)

        # JS should be symmetric
        assert np.isclose(result1.score, result2.score, rtol=0.05)

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

    def test_result_no_p_value(self, detector, baseline_data):
        """Test JS result doesn't include p-value."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert result.p_value is None

    def test_result_interpretation(self, detector, baseline_data):
        """Test result has meaningful interpretation."""
        detector.set_baseline(baseline_data)
        current_data = np.random.normal(loc=30, scale=10, size=1000)

        result = detector.detect(current_data)

        assert len(result.interpretation) > 0
        assert (
            "JS" in result.interpretation or "similar" in result.interpretation.lower()
        )
