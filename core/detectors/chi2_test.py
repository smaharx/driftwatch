"""
Chi-Squared Test Drift Detector

Chi-Squared test detects categorical distribution shifts.
- Use case: Product categories, regions, payment methods
- Output: χ² statistic and p-value
- Interpretation: p-value < 0.05 = significant drift
"""

import numpy as np
from scipy.stats import chi2_contingency

from .base import DriftDetector, DriftResult


class Chi2Detector(DriftDetector):
    """
    Chi-Squared Test detector.

    Tests if categorical distributions are significantly different.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Chi-Squared detector.

        Args:
            alpha: Significance level (default 0.05)
        """
        super().__init__(name="CHI2_TEST", threshold=alpha)
        self.alpha = alpha

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """
        Perform Chi-Squared test on categorical data.

        Args:
            current_data: Current categorical data

        Returns:
            DriftResult with χ² statistic and p-value
        """
        # Validate inputs
        self._validate_data(current_data)
        if self.reference_data is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        # Get unique categories
        categories = np.unique(np.concatenate([self.reference_data, current_data]))

        # Count occurrences in each category
        ref_counts = np.array(
            [np.sum(self.reference_data == cat) for cat in categories]
        )
        curr_counts = np.array([np.sum(current_data == cat) for cat in categories])

        # Build contingency table
        contingency_table = np.array([ref_counts, curr_counts])

        # Perform Chi-Squared test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        # Determine if drifted
        drifted = p_value < self.alpha

        # Create interpretation
        if chi2_stat < 1:
            chi2_interpretation = "χ² < 1 (similar distributions)"
        elif chi2_stat < 5:
            chi2_interpretation = "χ² < 5 (small difference)"
        elif chi2_stat < 10:
            chi2_interpretation = "χ² < 10 (moderate difference)"
        else:
            chi2_interpretation = "χ² ≥ 10 (large difference)"

        if p_value < 0.001:
            p_interpretation = "p < 0.001 (extremely strong evidence)"
        elif p_value < 0.05:
            p_interpretation = "p < 0.05 (strong evidence)"
        else:
            p_interpretation = "p ≥ 0.05 (weak evidence)"

        interpretation = (
            f"χ²={chi2_stat:.4f} ({chi2_interpretation}), {p_interpretation}"
        )

        if drifted:
            interpretation = "🚨 SIGNIFICANT DIFFERENCE - " + interpretation
        else:
            interpretation = "✅ NO SIGNIFICANT DIFFERENCE - " + interpretation

        return DriftResult(
            detector_name=self.name,
            score=chi2_stat,
            p_value=p_value,
            threshold=self.alpha,
            drifted=drifted,
            interpretation=interpretation,
        )
