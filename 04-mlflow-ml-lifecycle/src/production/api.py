from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.production.load_model import (
    load_production_model,
    predict_churn,
)


model = None


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = load_production_model()

    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    predictions, probabilities = predict_churn(
        model,
        input_df,
    )

    return {
        "prediction": int(predictions[0]),
        "churn_probability": float(probabilities[0]),
    }