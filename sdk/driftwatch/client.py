import pandas as pd
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    id: str
    name: str
    feature_names: list[str]
    model_type: str


class DriftClient:
    """
    DriftWatch Python SDK.

    Provides a simple interface for registering models,
    uploading baselines, and logging production data for drift analysis.
    """

    def __init__(self, api_url: str, timeout: int = 30) -> None:
        self._base_url = api_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def register_model(
        self,
        name: str,
        feature_names: list[str],
        model_type: str = "classification",
        description: Optional[str] = None,
    ) -> ModelInfo:
        """Register a new ML model for drift monitoring."""
        response = self._session.post(
            f"{self._base_url}/api/v1/models",
            json={
                "name": name,
                "feature_names": feature_names,
                "model_type": model_type,
                "description": description,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return ModelInfo(
            id=data["id"],
            name=data["name"],
            feature_names=data["feature_names"],
            model_type=data["model_type"],
        )

    def log_baseline(
        self,
        model_id: str,
        dataframe: pd.DataFrame,
        categorical_features: Optional[list[str]] = None,
    ) -> None:
        """
        Upload training data as the reference baseline.

        Args:
            model_id: UUID of the registered model.
            dataframe: Training DataFrame. All columns used as features.
            categorical_features: Column names to treat as categorical.
                                  All others treated as numerical.
        """
        categorical_features = categorical_features or []
        features = self._dataframe_to_features(dataframe, categorical_features)

        response = self._session.post(
            f"{self._base_url}/api/v1/models/{model_id}/baseline",
            json={"features": features},
            timeout=self._timeout,
        )
        response.raise_for_status()

    def log(
        self,
        model_id: str,
        dataframe: pd.DataFrame,
        categorical_features: Optional[list[str]] = None,
    ) -> dict:
        """
        Submit production data for drift analysis.

        Args:
            model_id: UUID of the registered model.
            dataframe: Production DataFrame batch to analyze.
            categorical_features: Column names to treat as categorical.

        Returns:
            Drift report dict with scores and drifted features.
        """
        categorical_features = categorical_features or []
        features = self._dataframe_to_features(dataframe, categorical_features)

        response = self._session.post(
            f"{self._base_url}/api/v1/models/{model_id}/runs",
            json={"features": features},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> list[dict]:
        """Fetch drift alerts with optional filtering."""
        params = {}
        if model_id:
            params["model_id"] = model_id
        if severity:
            params["severity"] = severity

        response = self._session.get(
            f"{self._base_url}/api/v1/alerts",
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def _dataframe_to_features(
        self,
        df: pd.DataFrame,
        categorical_features: list[str],
    ) -> list[dict]:
        """Convert a DataFrame to the API feature format."""
        features = []
        for col in df.columns:
            feature_type = "categorical" if col in categorical_features else "numerical"
            values = df[col].tolist()
            features.append(
                {
                    "feature_name": col,
                    "feature_type": feature_type,
                    "values": values,
                }
            )
        return features
