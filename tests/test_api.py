"""Tests for the FinGuard AI FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_check():
    """Verify that the health check endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "environment" in data


def test_get_transactions_default():
    """Verify listing transactions with default limit."""
    response = client.get("/api/transactions")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert "returned_records" in data
    assert "transactions" in data
    assert data["returned_records"] == 20
    assert len(data["transactions"]) == 20
    assert data["total_records"] == 100


def test_get_transactions_custom_limit():
    """Verify listing transactions with custom pagination limit."""
    response = client.get("/api/transactions?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["returned_records"] == 5
    assert len(data["transactions"]) == 5


def test_get_transactions_filter_risk_level():
    """Verify filtering transactions by risk rating."""
    response = client.get("/api/transactions?risk_level=HIGH&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["returned_records"] <= 10
    assert data["total_records"] > 0


def test_get_transactions_invalid_risk_level():
    """Verify that an invalid risk_level returns 400 Bad Request."""
    response = client.get("/api/transactions?risk_level=EXTREME")
    assert response.status_code == 400
    assert "Invalid risk_level" in response.json()["detail"]


def test_get_high_risk_transactions():
    """Verify retrieving high-risk transaction list sorted by risk score."""
    response = client.get("/api/transactions/high-risk?limit=5")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) == 5

    # Verify attributes and descending score sort
    for item in items:
        assert "transaction_id" in item
        assert "risk_score" in item
        assert "risk_level" in item
        assert item["risk_level"] in ["MEDIUM", "HIGH"]
        assert "triggered_rules" in item

    scores = [item["risk_score"] for item in items]
    assert scores == sorted(scores, reverse=True)


def test_get_transaction_by_id_success():
    """Verify looking up an existing transaction returns 200 and accurate fields."""
    response = client.get("/api/transactions/TX-1001")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-1001"
    assert data["customer_id"] == "CUST-101"
    assert "amount" in data
    assert "transaction_type" in data


def test_get_transaction_by_id_not_found():
    """Verify looking up a nonexistent transaction returns 404."""
    response = client.get("/api/transactions/TX-DOES-NOT-EXIST")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_transaction_risk_success():
    """Verify risk evaluation endpoint returns rules + ML assessment."""
    response = client.get("/api/transactions/TX-1001/risk")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-1001"
    assert "final_risk_level" in data
    assert "final_risk_score" in data
    assert "is_flagged" in data
    assert "ml_probability" in data
    assert "triggered_rules" in data
    assert "explanation" in data


def test_get_transaction_risk_not_found():
    """Verify risk evaluation on nonexistent transaction returns 404."""
    response = client.get("/api/transactions/TX-NONEXISTENT/risk")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_run_investigation_success():
    """Verify executing the AI investigation workflow on an existing transaction."""
    response = client.post("/api/investigate", json={"transaction_id": "TX-1003"})
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-1003"
    assert data["workflow_status"] == "COMPLETED"
    assert data["final_risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "investigation_summary" in data
    assert data["investigation_summary"] is not None
    assert len(data["investigation_summary"]) > 0
    assert "recommended_action" in data


def test_run_investigation_not_found():
    """Verify executing investigation on nonexistent transaction returns 404."""
    response = client.post("/api/investigate", json={"transaction_id": "TX-UNKNOWN"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_run_investigation_invalid_body():
    """Verify submitting an empty or invalid payload returns 422 Unprocessable Entity."""
    response = client.post("/api/investigate", json={})
    assert response.status_code == 422
