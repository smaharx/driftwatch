"""Unit tests for Chi-Squared detector."""

import numpy as np
import pytest
from core.detectors.chi2_test import Chi2Detector
from core.detectors.base import DriftResult


class TestChi2Detector:
    """Test Chi-Squared Test detector for categorical data."""

    @pytest.fixture
    def detector(self):
        """Create Chi-Squared detector instance."""
        return Chi2Detector(alpha=0.05)

    @pytest.fixture
    def baseline_categories(self):
        """Create baseline categorical data."""
        np.random.seed(42)
        # Distribution: 50% A, 30% B, 15% C, 5% D
        return np.random.choice(["A", "B", "C", "D"], size=1000, p=[0.5, 0.3, 0.15, 0.05])

    # === Basic Tests ===

    def test_detector_initialization(self, detector):
        """Test Chi² detector initializes correctly."""
        assert detector.name == "CHI2_TEST"
        assert detector.alpha == 0.05

    def test_set_baseline(self, detector, baseline_categories):
        """Test setting baseline categorical data."""
        detector.set_baseline(baseline_categories)
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 1000

    # === No Drift Tests ===

    def test_no_drift_identical_distribution(self, detector, baseline_categories):
        """Test Chi² detects no drift for identical distributions."""
        detector.set_baseline(baseline_categories)
        current_categories = baseline_categories.copy()

        result = detector.detect(current_categories)

        assert isinstance(result, DriftResult)
        assert result.drifted == False
        assert result.p_value > 0.05

    def test_no_drift_very_similar(self, detector, baseline_categories):
        """Test Chi² detects no drift for very similar distributions."""
        detector.set_baseline(baseline_categories)

        # Slightly different distribution
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.49, 0.31, 0.15, 0.05]
        )

        result = detector.detect(current_categories)

        assert result.p_value > 0.05

    # === Drift Detected Tests ===

    def test_drift_detected_changed_proportions(self, detector, baseline_categories):
        """Test Chi² detects drift with changed proportions."""
        detector.set_baseline(baseline_categories)

        # Major shift: 40% A, 40% B, 15% C, 5% D
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.4, 0.4, 0.15, 0.05]
        )

        result = detector.detect(current_categories)

        assert isinstance(result, DriftResult)
        assert result.drifted == True
        assert result.p_value < 0.05

    def test_major_drift_very_different(self, detector, baseline_categories):
        """Test Chi² detects major drift."""
        detector.set_baseline(baseline_categories)

        # Extreme shift
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.1, 0.1, 0.1, 0.7]
        )

        result = detector.detect(current_categories)

        assert result.drifted == True
        assert result.p_value < 0.001

    # === Statistical Properties Tests ===

    def test_chi2_statistic_non_negative(self, detector, baseline_categories):
        """Test Chi² statistic is always non-negative."""
        detector.set_baseline(baseline_categories)
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.5, 0.3, 0.15, 0.05]
        )

        result = detector.detect(current_categories)

        assert result.score >= 0

    def test_p_value_in_range(self, detector, baseline_categories):
        """Test p-value is between 0 and 1."""
        detector.set_baseline(baseline_categories)
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.5, 0.3, 0.15, 0.05]
        )

        result = detector.detect(current_categories)

        assert 0 <= result.p_value <= 1

    # === Error Handling Tests ===

    def test_detect_without_baseline_raises_error(self, detector):
        """Test detect() raises error if baseline not set."""
        current_data = np.array(["A", "B", "C"])

        with pytest.raises(ValueError, match="Baseline not set"):
            detector.detect(current_data)

    def test_detect_empty_current_raises_error(self, detector, baseline_categories):
        """Test empty current data raises ValueError."""
        detector.set_baseline(baseline_categories)

        with pytest.raises(ValueError, match="Data cannot be empty"):
            detector.detect(np.array([]))

    # === Result Validation ===

    def test_result_has_p_value(self, detector, baseline_categories):
        """Test DriftResult includes p_value."""
        detector.set_baseline(baseline_categories)
        current_categories = np.random.choice(
            ["A", "B", "C", "D"], size=1000, p=[0.5, 0.3, 0.15, 0.05]
        )

        result = detector.detect(current_categories)

        assert result.p_value != None

    # === Numeric Labels Test ===

    def test_numeric_categories(self, detector):
        """Test Chi² works with numeric categorical data."""
        baseline = np.array([0, 0, 1, 1, 2] * 200)  # 40% 0, 40% 1, 20% 2
        detector.set_baseline(baseline)

        current = np.array([0, 0, 1, 1, 2] * 200)
        result = detector.detect(current)

        assert isinstance(result, DriftResult)

