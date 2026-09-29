from pathlib import Path

from app.models import PredictionRequest
from app.services.predictor import SentimentPredictor


project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "artifacts" / "sentiment_model.joblib"

predictor = SentimentPredictor(model_path)


def test_predict_returns_valid_response() -> None:
    request = PredictionRequest(
        text="this application is very useful"
    )

    response = predictor.predict(request)

    assert response.label in {"positive", "negative"}
    assert 0.0 <= response.confidence <= 1.0