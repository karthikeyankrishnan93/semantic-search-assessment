from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from src import api
import numpy as np

class DummyModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[0.1] * 384 for _ in texts])

class DummyQdrantClient:
    def get_collection(self, collection_name):
        return SimpleNamespace(points_count=1)

    def query_points(self, **kwargs):
        result = SimpleNamespace(
            payload={
                "source": "banking_policy.txt",
                "page": 1,
                "section_heading": "Customer Service",
                "text": "Customer requests must be handled professionally."
            },
            score=0.85
        )
        return SimpleNamespace(points=[result])

@pytest.fixture
def test_client(monkeypatch):
    monkeypatch.setattr(api, "APP_API_KEY", "test-api-key")
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "client", DummyQdrantClient())

    return TestClient(api.app)

def test_health_api(test_client):
    response = test_client.get(
        "/health",
        headers={"x-api-key": "test-api-key"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_search_api(test_client):
    response = test_client.post(
        "/search",
        headers={"x-api-key": "test-api-key"},
        json={"query": "customer service", "top_k": 3}
    )

    assert response.status_code == 200

def test_search_result_structure(test_client):
    response = test_client.post(
        "/search",
        headers={"x-api-key": "test-api-key"},
        json={"query": "customer service", "top_k": 3}
    )

    result = response.json()["results"][0]

    assert result["source"] == "banking_policy.txt"
    assert result["page"] == 1
    assert result["score"] == 0.85
    assert "text" in result

def test_invalid_top_k(test_client):
    response = test_client.post(
        "/search",
        headers={"x-api-key": "test-api-key"},
        json={"query": "customer service", "top_k": 21}
    )

    assert response.status_code == 422

def test_missing_api_key(test_client):
    response = test_client.get("/health")

    assert response.status_code == 401

def test_wrong_api_key(test_client):
    response = test_client.get(
        "/health",
        headers={"x-api-key": "wrong-key"}
    )

    assert response.status_code == 401