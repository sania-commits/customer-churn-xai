from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


VALID_CUSTOMER = {
    "call_failure": 8,
    "complains": 1,
    "subscription_length": 30,
    "charge_amount": 3,
    "seconds_of_use": 1500,
    "frequency_of_use": 25,
    "frequency_of_sms": 10,
    "distinct_called_numbers": 15,
    "age_group": 2,
    "tariff_plan": 1,
    "status": 1,
    "age": 30,
    "customer_value": 500.0,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["threshold"] == 0.51


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_segment" in data
    assert "threshold" in data

    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]

    assert data["risk_segment"] in [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    assert data["threshold"] == 0.51


def test_invalid_complains_value():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["complains"] = 5

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_missing_required_feature():
    incomplete_customer = VALID_CUSTOMER.copy()
    incomplete_customer.pop("age")

    response = client.post(
        "/predict",
        json=incomplete_customer,
    )

    assert response.status_code == 422


def test_internal_prediction_error(monkeypatch):
    from src import api

    def mock_predict_proba(_):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(
        api.model,
        "predict_proba",
        mock_predict_proba,
    )

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction failed."
    }

