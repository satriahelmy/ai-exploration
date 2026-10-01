# Project 01 — Python AI API

## Goal

Learn how to turn a trained machine learning model into a tested,
containerized, and reusable API service.

## Learning Flow

Training
→ Model Artifact
→ Inference Service
→ Pydantic Validation
→ FastAPI
→ Testing
→ Python Packaging
→ Docker
→ Continuous Integration

---

## Progress

- [x] 01. Train a Simple ML Model
- [x] 02. Save the Model Artifact
- [x] 03. Build an Inference Service
- [x] 04. Add Pydantic Models
- [x] 05. Expose the Model with FastAPI
- [x] 06. Add Predictor and API Tests
- [x] 07. Package the Python Project
- [x] 08. Containerize with Docker
- [x] 09. Add GitHub Actions CI
- [x] 10. Final Documentation

---

## Step 01 — Train a Simple ML Model

### Goal

Create a small machine learning model that can later be served through an API.

### What We Built

A sentiment classifier using:

- TF-IDF for text representation
- Logistic Regression for classification
- scikit-learn Pipeline to combine preprocessing and prediction

### Key Takeaway

Model training is only the beginning of an ML system.

---

## Step 02 — Save the Model Artifact

### Goal

Separate model training from model inference.

### What We Built

The trained pipeline was serialized into:

`artifacts/sentiment_model.joblib`

### Key Takeaway

Applications should load an existing model artifact instead of retraining the model every time they start.

---

## Step 03 — Build an Inference Service

### Goal

Create a dedicated layer responsible for loading the model and making predictions.

### What We Built

A predictor service under:

`app/services/predictor.py`

### Key Takeaway

Inference logic should be separated from API routes.

---

## Step 04 — Add Pydantic Models

### Goal

Define clear input and output contracts for the application.

### What We Learned

Pydantic provides:

- request validation
- response schemas
- type safety
- automatic API documentation integration

### Key Takeaway

An API should have an explicit contract rather than accepting arbitrary data.

---

## Step 05 — Expose the Model with FastAPI

### Goal

Make the model accessible through HTTP.

### What We Built

A FastAPI application that receives text, sends it to the predictor,
and returns the prediction.

### Key Takeaway

FastAPI acts as the interface between external applications and the ML inference layer.

---

## Step 06 — Add Predictor and API Tests

### Goal

Verify both the prediction logic and HTTP interface.

### What We Tested

- predictor behavior
- valid API requests
- response structure
- invalid input handling

### Key Takeaway

Testing the model alone is not enough. The surrounding application behavior also needs tests.

---

## Step 07 — Package the Python Project

### Goal

Make dependencies and project configuration reproducible.

### What We Built

A `pyproject.toml` containing application and development dependencies.

### Important Lesson

The scikit-learn version used to load a serialized model should be compatible
with the version used to create the artifact.

### Key Takeaway

Model artifacts have software dependencies too.

---

## Step 08 — Containerize with Docker

### Goal

Package the application and its runtime environment into a container.

### What We Built

- `Dockerfile`
- `.dockerignore`

### Key Takeaway

Docker reduces differences between development and deployment environments.

---

## Step 09 — Add GitHub Actions CI

### Goal

Automatically validate the project whenever code changes.

### What We Built

A GitHub Actions workflow that runs the project tests.

### Key Takeaway

Continuous Integration turns testing from a manual action into part of the development workflow.

---

## Step 10 — Final Documentation

### Final Architecture

Client
→ FastAPI
→ Predictor Service
→ Trained Model Artifact

### Final Lesson

The biggest lesson from this project was that production ML involves much more
than `model.fit()`.

A usable ML system also needs application architecture, validation, testing,
dependency management, containerization, and automated checks.