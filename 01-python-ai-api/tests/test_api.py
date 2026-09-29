from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict_endpoint() -> None:
    response = client.post(
        "/predict",
        json={
            "text": "this application is very useful"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "confidence" in data

    assert data["label"] in {"positive", "negative"}
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_empty_text() -> None:
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_text() -> None:
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422