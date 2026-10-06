import mlflow
import mlflow.sklearn

from src.production.config import PREDICTION_THRESHOLD


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/telco-churn-model@champion"


def load_production_model():
    return mlflow.sklearn.load_model(MODEL_URI)


def predict_churn(model, data):
    probabilities = model.predict_proba(data)[:, 1]

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    return predictions, probabilities