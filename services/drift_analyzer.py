from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from core.config import settings
from core.detectors.chi2_test import Chi2Detector
from core.detectors.ks_test import KSTestDetector
from core.detectors.psi import PSIDetector
from db.models import Alert, Baseline, MLModel, Run

logger = logging.getLogger(__name__)


def _make_json_serializable(obj: any) -> any:
    """
    Recursively convert all non-JSON-safe types to Python natives.

    Handles: numpy scalars, numpy arrays, inf, nan, bool_, int_, float64.
    PostgreSQL JSON columns reject all of these.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (np.isinf(f) or np.isnan(f)) else f
    if isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    if isinstance(obj, float):
        return None if (np.isinf(obj) or np.isnan(obj)) else obj
    if isinstance(obj, bool):
        return obj
    return obj


def _compute_severity(score: float, threshold: float) -> str:
    """
    Derive alert severity from how far the score exceeds the threshold.

    - high   : score > 2× threshold
    - medium : score > 1.5× threshold
    - low    : score > threshold
    """
    ratio = score / threshold if threshold > 0 else 0
    if ratio > 2.0:
        return "high"
    if ratio > 1.5:
        return "medium"
    return "low"


def _profile_feature(values: list) -> dict:
    """
    Compute summary statistics for a numerical feature.
    Stored in the baseline so we never need the raw data again.
    """
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "count": len(arr),
    }


def _build_distribution(values: list, feature_type: str) -> dict:
    """
    Build a histogram representation of the feature distribution.
    This is what the drift detectors compare against at run time.
    """
    if feature_type == "categorical":
        unique, counts = np.unique(values, return_counts=True)
        return {
            "bins": unique.tolist(),
            "counts": counts.tolist(),
        }

    arr = np.array(values, dtype=float)
    counts, bin_edges = np.histogram(arr, bins=10)
    return {
        "bins": bin_edges.tolist(),
        "counts": counts.tolist(),
    }


class DriftAnalyzer:
    """
    Orchestrates drift detection for a single analysis run.

    Loads baselines from the database, runs all applicable detectors
    against the submitted production data, persists results, and
    creates Alert records for any features that exceed thresholds.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Public interface ───────────────────────────────────────────────────

    def store_baseline(
        self,
        model: MLModel,
        features: list[dict],
    ) -> list[Baseline]:
        """
        Persist reference distributions for all submitted features.

        Replaces any existing baseline for the same model+feature pair
        so re-uploading baseline data is idempotent.
        """
        stored: list[Baseline] = []

        for feature in features:
            name = feature["feature_name"]
            ftype = feature["feature_type"]
            values = feature["values"]

            # Remove stale baseline for this feature if it exists
            existing = (
                self._db.query(Baseline)
                .filter(
                    Baseline.model_id == model.id,
                    Baseline.feature_name == name,
                )
                .first()
            )
            if existing:
                self._db.delete(existing)

            stats = _profile_feature(values) if ftype == "numerical" else {}
            distribution = _build_distribution(values, ftype)

            baseline = Baseline(
                model_id=model.id,
                feature_name=name,
                feature_type=ftype,
                statistics=stats,
                distribution=distribution,
                sample_size=len(values),
            )
            self._db.add(baseline)
            stored.append(baseline)

        self._db.commit()
        logger.info("Stored %d baselines for model %s", len(stored), model.id)
        return stored

    def analyze(self, run: Run, features: list[dict]) -> Run:
        """
        Run drift detection for all submitted features against stored baselines.

        Updates the Run record in place with results and status.
        Creates Alert records for every drifted feature.
        """
        run.status = "running"
        self._db.commit()

        try:
            results, drifted_features, all_scores = self._run_detectors(
                run=run,
                features=features,
            )

            overall_score = float(np.mean(all_scores)) if all_scores else 0.0

            clean_results = _make_json_serializable(results)
            clean_score = (
                float(overall_score)
                if overall_score is not None
                and not (np.isinf(overall_score) or np.isnan(overall_score))
                else 0.0
            )

            run.status = "completed"
            run.drift_results = clean_results
            run.overall_drift_score = clean_score
            run.drifted_features = drifted_features
            run.completed_at = datetime.now(timezone.utc)
            
            if drifted_features:
                from services.notifications import send_drift_alert
                send_drift_alert(
                    model_id=run.model_id,
                    run_id=run.id,
                    drifted_features=drifted_features,
                    overall_score=clean_score,
                )

        except Exception as exc:
            logger.exception("Drift analysis failed for run %s", run.id)
            run.status = "failed"
            run.error_message = str(exc)
            run.completed_at = datetime.now(timezone.utc)

        self._db.commit()
        self._db.refresh(run)
        return run

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_detectors(
        self,
        run: Run,
        features: list[dict],
    ) -> tuple[dict, list[str], list[float]]:
        """
        Iterate over submitted features, load their baselines, run detectors.

        Returns:
            results          — per-feature detector output (stored as JSON)
            drifted_features — names of features that drifted
            all_scores       — drift scores used to compute overall score
        """
        results: dict = {}
        drifted_features: list[str] = []
        all_scores: list[float] = []

        for feature in features:
            name = feature["feature_name"]
            ftype = feature["feature_type"]
            values = feature["values"]

            baseline = (
                self._db.query(Baseline)
                .filter(
                    Baseline.model_id == run.model_id,
                    Baseline.feature_name == name,
                )
                .first()
            )

            if not baseline:
                logger.warning(
                    "No baseline found for feature '%s' on model %s — skipping",
                    name,
                    run.model_id,
                )
                results[name] = {"error": "no baseline found"}
                continue

            feature_result = self._detect_feature(
                run=run,
                feature_name=name,
                feature_type=ftype,
                current_values=values,
                baseline=baseline,
            )

            results[name] = feature_result
            all_scores.append(feature_result["score"])

            if feature_result["drifted"]:
                drifted_features.append(name)

        return results, drifted_features, all_scores

    def _detect_feature(
        self,
        run: Run,
        feature_name: str,
        feature_type: str,
        current_values: list,
        baseline: Baseline,
    ) -> dict:
        """
        Select and run the appropriate detector for a single feature.

        Numerical  → PSI (primary) + KS test (secondary)
        Categorical → Chi-Squared
        """
        reference = np.array(baseline.distribution["counts"], dtype=float)

        if feature_type == "categorical":
            return self._run_chi2(
                run=run,
                feature_name=feature_name,
                current_values=current_values,
                baseline=baseline,
                reference=reference,
            )

        return self._run_psi_ks(
            run=run,
            feature_name=feature_name,
            current_values=current_values,
            baseline=baseline,
        )

    def _run_psi_ks(
        self,
        run: Run,
        feature_name: str,
        current_values: list,
        baseline: Baseline,
    ) -> dict:
        """Run PSI + KS test on a numerical feature."""
        reference_values = np.array(baseline.distribution["bins"][:-1], dtype=float)
        current_arr = np.array(current_values, dtype=float)

        # PSI
        psi = PSIDetector()
        psi.set_baseline(reference_values)
        psi_result = psi.detect(current_arr)

        # KS test
        ks = KSTestDetector()
        ks.set_baseline(reference_values)
        ks_result = ks.detect(current_arr)

        drifted = psi_result.drifted or ks_result.drifted
        score = psi_result.score

        if drifted:
            self._create_alert(
                run=run,
                feature_name=feature_name,
                detector_type="PSI",
                drift_score=score,
                threshold=settings.PSI_THRESHOLD,
            )

        return {
            "feature_type": "numerical",
            "drifted": drifted,
            "score": score,
            "psi": psi_result.to_dict(),
            "ks": ks_result.to_dict(),
        }

    def _run_chi2(
        self,
        run: Run,
        feature_name: str,
        current_values: list,
        baseline: Baseline,
        reference: np.ndarray,
    ) -> dict:
        """Run Chi-Squared test on a categorical feature."""
        current_arr = np.array(current_values)

        chi2 = Chi2Detector()
        chi2.set_baseline(np.array(baseline.distribution["bins"]))
        chi2_result = chi2.detect(current_arr)

        if chi2_result.drifted:
            self._create_alert(
                run=run,
                feature_name=feature_name,
                detector_type="CHI2",
                drift_score=chi2_result.score,
                threshold=settings.KS_ALPHA,
            )

        return {
            "feature_type": "categorical",
            "drifted": chi2_result.drifted,
            "score": chi2_result.score,
            "chi2": chi2_result.to_dict(),
        }

    def _create_alert(
        self,
        run: Run,
        feature_name: str,
        detector_type: str,
        drift_score: float,
        threshold: float,
    ) -> None:
        """Persist an Alert record for a drifted feature."""
        severity = _compute_severity(drift_score, threshold)
        alert = Alert(
            run_id=run.id,
            model_id=run.model_id,
            feature_name=feature_name,
            detector_type=detector_type,
            drift_score=float(drift_score),
            threshold=float(threshold),
            severity=severity,
        )
        self._db.add(alert)
        logger.info(
            "Alert created — feature=%s detector=%s score=%.4f severity=%s",
            feature_name,
            detector_type,
            drift_score,
            severity,
        )
