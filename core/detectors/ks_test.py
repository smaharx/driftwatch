"""
KS Test (Kolmogorov-Smirnov) Drift Detector

KS Test determines statistical significance of distribution shift.
- Output: p-value
- Interpretation: p-value < 0.05 = significant drift
- Use case: Make binary alert decisions
"""

import numpy as np
from scipy.stats import ks_2samp

from .base import DriftDetector, DriftResult


class KSTestDetector(DriftDetector):
    """
    Kolmogorov-Smirnov Test detector.

    Tests if two distributions are statistically significantly different.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize KS Test detector.

        Args:
            alpha: Significance level (default 0.05)
        """
        super().__init__(name="KS_TEST", threshold=alpha)
        self.alpha = alpha

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """
        Perform KS Test between reference and current data.

        Args:
            current_data: New data to analyze

        Returns:
            DriftResult with KS statistic, p-value, and interpretation
        """
        # Validate inputs
        self._validate_data(current_data)
        if self.reference_data is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        # Perform KS test
        ks_statistic, p_value = ks_2samp(self.reference_data, current_data)

        # Determine if drifted (based on p-value)
        drifted = p_value < self.alpha

        # Create interpretation
        if p_value < 0.001:
            p_interpretation = "p < 0.001 (extremely strong evidence)"
        elif p_value < 0.05:
            p_interpretation = "p < 0.05 (strong evidence)"
        elif p_value < 0.1:
            p_interpretation = "p < 0.1 (weak evidence)"
        else:
            p_interpretation = "p ≥ 0.1 (no strong evidence)"

        if ks_statistic < 0.05:
            ks_interpretation = "tiny distance (KS < 0.05)"
        elif ks_statistic < 0.15:
            ks_interpretation = "small distance (KS < 0.15)"
        elif ks_statistic < 0.3:
            ks_interpretation = "moderate distance (KS < 0.3)"
        else:
            ks_interpretation = "large distance (KS ≥ 0.3)"

        interpretation = (
            f"KS={ks_statistic:.4f} ({ks_interpretation}), {p_interpretation}"
        )

        if drifted:
            interpretation = "🚨 SIGNIFICANT DIFFERENCE - " + interpretation
        else:
            interpretation = "✅ NO SIGNIFICANT DIFFERENCE - " + interpretation

        return DriftResult(
            detector_name=self.name,
            score=ks_statistic,
            p_value=p_value,
            threshold=self.alpha,
            drifted=drifted,
            interpretation=interpretation,
        )
