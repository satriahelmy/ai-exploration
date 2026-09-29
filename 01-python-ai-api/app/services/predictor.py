from pathlib import Path

import joblib


class SentimentPredictor:
    """Load a trained sentiment model and perform predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> dict[str, str | float]:
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]

        confidence = max(probabilities)

        return {
            "label": prediction,
            "confidence": float(confidence),
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "artifacts" / "sentiment_model.joblib"

    predictor = SentimentPredictor(model_path)

    result = predictor.predict("this application is very useful")

    print(result)