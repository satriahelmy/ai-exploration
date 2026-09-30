from fastapi.testclient import TestClient

from app.exceptions import LLMServiceError
from app.main import app, chat_service


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_chat_endpoint(monkeypatch):
    def fake_chat(message, history):
        return "This is a mocked response."

    monkeypatch.setattr(
        chat_service,
        "chat",
        fake_chat,
    )

    response = client.post(
        "/chat",
        json={
            "message": "What is machine learning?",
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "This is a mocked response."
    }


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={
            "message": "",
            "history": [],
        },
    )

    assert response.status_code == 422


def test_chat_handles_llm_failure(monkeypatch):
    def fake_chat(message, history):
        raise LLMServiceError(
            "Provider failed."
        )

    monkeypatch.setattr(
        chat_service,
        "chat",
        fake_chat,
    )

    response = client.post(
        "/chat",
        json={
            "message": "Hello",
            "history": [],
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The AI service is temporarily unavailable."
    }