"""
Jensen-Shannon Divergence Drift Detector

Jensen-Shannon measures flexible, symmetric divergence.
- Range: 0 to 1 (bounded and interpretable)
- Use case: Any distribution type, continuous or categorical
- Advantage: Symmetric (JSD(P||Q) = JSD(Q||P))
"""

import numpy as np
from scipy.spatial.distance import jensenshannon

from .base import DriftDetector, DriftResult


class JensenShannonDetector(DriftDetector):
    """
    Jensen-Shannon Divergence detector.

    Symmetric measure of distribution difference.
    Works for any distribution type.
    """

    def __init__(self, bins: int = 10, threshold: float = 0.15):
        """
        Initialize Jensen-Shannon detector.

        Args:
            bins: Number of bins for continuous data (default 10)
            threshold: Alert threshold (default 0.15)
        """
        super().__init__(name="JENSEN_SHANNON", threshold=threshold)
        self.bins = bins

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """
        Calculate Jensen-Shannon divergence.

        Args:
            current_data: New data to analyze

        Returns:
            DriftResult with JS divergence score
        """
        # Validate inputs
        self._validate_data(current_data)
        if self.reference_data is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        # Calculate JS divergence
        js_score = self._calculate_jensen_shannon(
            self.reference_data, current_data, self.bins
        )

        # Determine if drifted
        drifted = js_score > self.threshold

        # Create interpretation
        if js_score < 0.05:
            interpretation = "✅ Very similar (JS < 0.05)"
        elif js_score < 0.1:
            interpretation = "✅ Similar (JS < 0.1)"
        elif js_score < 0.2:
            interpretation = "⚠️  Moderate difference (0.1 ≤ JS < 0.2)"
        elif js_score < 0.5:
            interpretation = "⚠️  Large difference (0.2 ≤ JS < 0.5)"
        else:
            interpretation = "🚨 VERY LARGE DIFFERENCE (JS ≥ 0.5)"

        if drifted:
            interpretation = "🚨 DRIFT DETECTED - " + interpretation
        else:
            interpretation = "✅ NO DRIFT - " + interpretation

        return DriftResult(
            detector_name=self.name,
            score=js_score,
            p_value=None,  # JS doesn't have p-value
            threshold=self.threshold,
            drifted=drifted,
            interpretation=interpretation,
        )

    @staticmethod
    def _calculate_jensen_shannon(
        reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            JS divergence (0 to 1)
        """
        # Create bins from reference
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        # Histogram counts
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        curr_counts = np.histogram(current, bins=breakpoints)[0]

        # Convert to probabilities
        ref_probs = ref_counts / np.sum(ref_counts)
        curr_probs = curr_counts / np.sum(curr_counts)

        # Handle zeros
        epsilon = 1e-10
        ref_probs = np.where(ref_probs == 0, epsilon, ref_probs)
        curr_probs = np.where(curr_probs == 0, epsilon, curr_probs)

        # Normalize
        ref_probs = ref_probs / np.sum(ref_probs)
        curr_probs = curr_probs / np.sum(curr_probs)

        # Calculate JS divergence
        js = jensenshannon(ref_probs, curr_probs)

        return float(js)
