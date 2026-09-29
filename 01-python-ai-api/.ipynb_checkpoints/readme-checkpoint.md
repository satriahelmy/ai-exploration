# Python AI API

A simple machine learning API built to explore the fundamentals of production-style AI engineering.

This project demonstrates how to separate model training from inference, validate application data, expose a trained model through an HTTP API, and test the application automatically.

## Architecture

```text
Training Data
     |
     v
TF-IDF + Logistic Regression
     |
     v
Model Artifact
     |
     v
Sentiment Predictor
     |
     v
FastAPI
     |
     v
REST API
```

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── services/
│       ├── __init__.py
│       └── predictor.py
├── artifacts/
│   └── sentiment_model.joblib
├── training/
│   └── train.py
├── tests/
│   ├── test_api.py
│   └── test_predictor.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Tech Stack

- Python
- scikit-learn
- Pydantic
- FastAPI
- Uvicorn
- pytest

## Setup

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

## Train the Model

```bash
python training/train.py
```

The training script creates the model artifact:

```text
artifacts/sentiment_model.joblib
```

## Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Predict Sentiment

```http
POST /predict
```

Example request:

```json
{
  "text": "this application is very useful"
}
```

Example response:

```json
{
  "label": "positive",
  "confidence": 0.5875
}
```

## Run Tests

```bash
python -m pytest -v
```

The test suite covers:

- health endpoint
- prediction endpoint
- request validation
- missing input validation
- predictor output validation

## Learning Goals

This project explores:

- separating model training from inference
- saving and loading model artifacts
- structuring a Python application
- Python type hints
- request and response schemas with Pydantic
- input validation
- serving ML models with FastAPI
- REST API fundamentals
- automated testing with pytest
- Python project and dependency management

## Notes

The sentiment model is intentionally trained on a very small dataset.

The goal of this project is not to build a high-quality sentiment classifier, but to understand the engineering workflow required to turn a trained machine learning model into a structured, testable API service.