"""
PSI (Population Stability Index) Drift Detector

PSI measures the magnitude of distribution shift.
- Range: 0 to ∞
- Interpretation: PSI > 0.25 = significant drift
- Use case: Track drift magnitude over time
"""


import numpy as np

from .base import DriftDetector, DriftResult


class PSIDetector(DriftDetector):
    """
    Population Stability Index detector.

    Measures how much a distribution has shifted from baseline.
    """

    def __init__(self, bins: int = 10, threshold: float = 0.25):
        """
        Initialize PSI detector.

        Args:
            bins: Number of bins for histogram (default 10)
            threshold: Alert threshold (default 0.25)
        """
        super().__init__(name="PSI", threshold=threshold)
        self.bins = bins

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """
        Calculate PSI between reference and current data.

        Args:
            current_data: New data to analyze

        Returns:
            DriftResult with PSI score and interpretation
        """
        # Validate inputs
        self._validate_data(current_data)
        if self.reference_data is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        # Calculate PSI
        psi_score = self._calculate_psi(self.reference_data, current_data, self.bins)

        # Determine if drifted
        drifted = psi_score > self.threshold

        # Create interpretation
        if psi_score < 0.05:
            interpretation = "✅ No drift (PSI < 0.05)"
        elif psi_score < 0.15:
            interpretation = "⚠️  Small drift (0.05 ≤ PSI < 0.15)"
        elif psi_score < 0.25:
            interpretation = "⚠️  Moderate drift (0.15 ≤ PSI < 0.25)"
        else:
            interpretation = "🚨 SIGNIFICANT DRIFT (PSI ≥ 0.25)"

        return DriftResult(
            detector_name=self.name,
            score=psi_score,
            p_value=None,  # PSI doesn't have p-value
            threshold=self.threshold,
            drifted=drifted,
            interpretation=interpretation,
        )

    @staticmethod
    def _calculate_psi(
        reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:

        # Create bins from reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Histogram counts
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        curr_counts = np.histogram(current, bins=breakpoints)[0]

        # Convert to percentages
        ref_pct = ref_counts / np.sum(ref_counts)
        curr_pct = curr_counts / np.sum(curr_counts)

        # Handle zero proportions (add small epsilon)
        epsilon = 1e-10
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)

        # Re-normalize after epsilon adjustment
        ref_pct = ref_pct / np.sum(ref_pct)
        curr_pct = curr_pct / np.sum(curr_pct)

        # Calculate PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

        # Handle NaN (from extreme outliers)
        if np.isnan(psi):
            psi = float("inf")  # Extreme drift returns infinity

        return float(psi)
