from pathlib import Path

from fastapi import FastAPI

from app.models import PredictionRequest, PredictionResponse
from app.services.predictor import SentimentPredictor


# Create the FastAPI application
app = FastAPI(
    title="Sentiment Prediction API",
    description="A simple API for sentiment classification.",
    version="1.0.0",
)


# Load the trained model
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "artifacts" / "sentiment_model.joblib"

predictor = SentimentPredictor(model_path)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Sentiment Prediction API is running."
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return predictor.predict(request)