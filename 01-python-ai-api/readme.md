# Python AI API

A small end-to-end machine learning API built to explore the fundamentals of AI engineering.

The project demonstrates how a trained machine learning model can be turned into a structured, validated, tested, and containerized API service.

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
Pydantic Validation
     |
     v
FastAPI
     |
     v
REST API
```

The application is packaged as a Docker image and validated automatically using GitHub Actions.

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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Tech Stack

- Python 3.11
- scikit-learn
- Pydantic
- FastAPI
- Uvicorn
- pytest
- Docker
- GitHub Actions

## Local Setup

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

## Train the Model

Run:

```bash
python training/train.py
```

The training pipeline uses TF-IDF and Logistic Regression and saves the trained pipeline to:

```text
artifacts/sentiment_model.joblib
```

The included model is intentionally trained on a very small dataset because the focus of this project is the engineering workflow rather than model performance.

## Run the API Locally

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

The automated tests cover:

- health endpoint behavior
- prediction endpoint behavior
- empty input validation
- missing input validation
- predictor response structure

## Docker

Build the Docker image:

```bash
docker build -t python-ai-api .
```

Run the container:

```bash
docker run -d \
  --name python-ai-api-container \
  -p 8000:8000 \
  python-ai-api
```

The containerized API will be available at:

```text
http://127.0.0.1:8000
```

Check running containers:

```bash
docker ps
```

View application logs:

```bash
docker logs python-ai-api-container
```

Stop the container:

```bash
docker stop python-ai-api-container
```

## Continuous Integration

GitHub Actions automatically validates changes to this project.

For each relevant push or pull request, the CI workflow:

1. creates a clean Ubuntu environment
2. installs Python 3.11 and project dependencies
3. runs the automated test suite
4. verifies that the Docker image can be built successfully

This helps ensure that the application works outside the local development environment.

## Learning Goals

This project explores:

- separating model training from inference
- persisting and loading ML model artifacts
- structuring a Python application
- dependency and package management with `pyproject.toml`
- request and response validation with Pydantic
- serving machine learning models through FastAPI
- REST API fundamentals
- automated testing with pytest
- containerization with Docker
- dependency version consistency between training and inference
- continuous integration with GitHub Actions

## What I Learned

Building a machine learning model is only one part of an AI system.

A usable AI service also requires clear boundaries between training and inference, validated inputs and outputs, reproducible dependencies, automated testing, containerization, and automated integration checks.

This project intentionally uses a simple model so the engineering lifecycle remains the primary focus.