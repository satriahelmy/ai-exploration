from pathlib import Path

import joblib

from app.models import PredictionRequest, PredictionResponse


class SentimentPredictor:
    """Load a trained sentiment model and perform predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        prediction = self.model.predict([request.text])[0]
        probabilities = self.model.predict_proba([request.text])[0]

        confidence = float(max(probabilities))

        return PredictionResponse(
            label=prediction,
            confidence=confidence,
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "artifacts" / "sentiment_model.joblib"

    predictor = SentimentPredictor(model_path)

    request = PredictionRequest(
        text="this application is very useful"
    )

    result = predictor.predict(request)

    print(result)