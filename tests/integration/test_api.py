import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200


def test_register_model(client: TestClient, sample_model_payload: dict):
    response = client.post("/api/v1/models", json=sample_model_payload)
    assert response.status_code in (201, 409)


def test_list_models(client: TestClient):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_alerts(client: TestClient):
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
