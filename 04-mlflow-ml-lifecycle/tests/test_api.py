from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient


mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array(
    [[0.10, 0.90]]
)


@patch(
    "src.production.api.load_production_model",
    return_value=mock_model,
)
def test_health(mock_load_model):
    from src.production.api import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch(
    "src.production.api.load_production_model",
    return_value=mock_model,
)
def test_predict(mock_load_model):
    from src.production.api import app

    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 8,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 716.0,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] == 1
    assert body["churn_probability"] == 0.9

@patch(
    "src.production.api.load_production_model",
    return_value=mock_model,
)
def test_predict_invalid_request(mock_load_model):
    from src.production.api import app

    invalid_payload = {
        "gender": "Female",
        "SeniorCitizen": 0
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_payload,
        )

    assert response.status_code == 422