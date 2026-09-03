import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_model_payload():
    return {
        "name": "test-model",
        "description": "Test model for integration tests",
        "model_type": "classification",
        "feature_names": ["age", "income", "region"],
    }