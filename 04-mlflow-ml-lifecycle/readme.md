# MLflow & ML Lifecycle

An end-to-end machine learning lifecycle project for predicting customer churn using the IBM Telco Customer Churn dataset.

This project focuses not only on training a machine learning model, but also on the engineering workflow around it: experiment tracking, model evaluation, model registry, reproducible training, API serving, containerization, testing, and continuous integration.

## Architecture

```text
Telco Customer Churn Dataset
            |
            v
     Data Preparation
            |
            v
      Model Training
            |
            v
   MLflow Experiment Tracking
     |      |       |
   Params Metrics Artifacts
            |
            v
      Model Evaluation
            |
            v
      Model Registry
            |
            v
   @champion Model Alias
            |
            v
      FastAPI Inference
            |
            v
       Docker Compose
       |            |
       v            v
   FastAPI        MLflow
       |
       v
   Prediction API
```

## Project Goals

The main goals of this project are to learn how to:

- build a reproducible machine learning training pipeline
- track experiments using MLflow
- compare models and hyperparameters
- log parameters, metrics, artifacts, and models
- evaluate model performance using validation and test sets
- register and version models
- manage production models using MLflow aliases
- load registered models for inference
- expose predictions through FastAPI
- test the inference API
- containerize MLflow and FastAPI with Docker Compose
- run automated tests using GitHub Actions

## Dataset

This project uses the IBM Telco Customer Churn dataset.

The dataset contains 7,043 customer records and includes information such as:

- customer demographics
- subscription tenure
- internet and phone services
- contract type
- payment method
- monthly charges
- total charges
- customer churn status

The target variable is:

```text
Churn
```

The dataset is imbalanced:

```text
No     ~73%
Yes    ~27%
```

## Data Preparation

The preprocessing pipeline includes:

- removing `customerID`
- converting `TotalCharges` to numeric
- handling missing `TotalCharges`
- scaling numerical features using `StandardScaler`
- encoding categorical features using `OneHotEncoder`
- stratified train, validation, and test splitting

The data is separated into:

```text
Train
Validation
Test
```

The validation set is used for model and threshold selection.

The test set remains untouched until final evaluation.

## Models

Several models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

Logistic Regression provided the strongest balance between performance, simplicity, and interpretability for this dataset.

Further experiments included:

- hyperparameter tuning
- class weighting
- decision threshold tuning
- SMOTE
- random undersampling
- cross-validation

The selected model configuration is:

```text
Model: Logistic Regression
C: 0.1
Class Weight: balanced
Max Iterations: 1000
Prediction Threshold: 0.55
```

## Final Test Performance

The final model was evaluated once on the untouched test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.7559 |
| Precision | 0.5285 |
| Recall | 0.7433 |
| F1 Score | 0.6178 |
| ROC-AUC | 0.8414 |

The model prioritizes churn recall, which allows it to identify a larger proportion of customers who may churn.

## Cross-Validation

The selected Logistic Regression configuration was also evaluated using 5-fold cross-validation.

| Metric | Mean | Std |
|---|---:|---:|
| Precision | 0.5204 | 0.0147 |
| Recall | 0.7987 | 0.0143 |
| F1 Score | 0.6300 | 0.0119 |
| ROC-AUC | 0.8459 | 0.0073 |

The relatively small standard deviations indicate stable performance across folds.

## Experiment Tracking with MLflow

MLflow is used to track the machine learning experiments.

Tracked information includes:

### Parameters

Examples:

```text
model
C
class_weight
n_estimators
learning_rate
max_depth
threshold
```

### Metrics

Examples:

```text
accuracy
precision
recall
f1
roc_auc
```

### Artifacts

Examples include:

- confusion matrices
- classification reports
- trained model artifacts

MLflow UI makes it possible to compare experiments and inspect model performance across runs.

## Model Registry

The selected production model is registered in MLflow as:

```text
telco-churn-model
```

Production code loads the model using the alias:

```text
models:/telco-churn-model@champion
```

Using an alias decouples the application from a specific model version.

Instead of changing application code when a new model becomes production-ready, the `champion` alias can be moved to another registered model version.

## Production Training

Experiment scripts are kept separate from the production training pipeline.

```text
src/
├── data.py
├── preprocessing.py
├── ...
└── production/
    ├── config.py
    ├── train.py
    ├── load_model.py
    ├── smoke_test.py
    └── api.py
```

Final model configuration is centralized in:

```text
src/production/config.py
```

This separates exploratory experimentation from reproducible production training.

## FastAPI Inference

The registered champion model is exposed through a FastAPI application.

Start the API locally:

```bash
uvicorn src.production.api:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

Example response:

```json
{
  "prediction": 1,
  "churn_probability": 0.8906856783317377
}
```

## Docker Compose

The production services can run using Docker Compose.

```text
Docker Compose
|
├── mlflow
|   ├── Tracking Server
|   ├── Model Registry
|   ├── mlflow.db
|   └── mlartifacts/
|
└── api
    ├── FastAPI
    └── champion model inference
```

Start the services:

```bash
docker compose up --build
```

Services are available at:

```text
MLflow UI
http://localhost:5000

FastAPI
http://localhost:8000

Swagger UI
http://localhost:8000/docs
```

Inside the Docker network, the API communicates with MLflow using:

```text
http://mlflow:5000
```

rather than `localhost`, because each container has its own network namespace.

## Testing

API tests cover:

- health endpoint
- valid prediction request
- invalid prediction request

Run the tests:

```bash
python -m pytest tests
```

MLflow model loading is mocked during API testing so the tests do not depend on a running MLflow server.

## Continuous Integration

GitHub Actions runs automated tests on pushes and pull requests.

The CI pipeline performs:

```text
Checkout Repository
        |
        v
Set Up Python 3.11
        |
        v
Install Dependencies
        |
        v
Run Pytest
```

This verifies that the application and tests can run in a clean environment independent of the local development machine.

## Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

### 3. Train the Production Model

```bash
python -m src.production.train
```

### 4. Run the Smoke Test

```bash
python -m src.production.smoke_test
```

### 5. Start the API

```bash
uvicorn src.production.api:app --reload
```

Alternatively, run the application stack with Docker:

```bash
docker compose up --build
```

## Key Lessons

This project demonstrates that building a machine learning system involves much more than calling `model.fit()`.

A complete ML lifecycle also requires:

```text
Data
  ↓
Experiments
  ↓
Evaluation
  ↓
Model Selection
  ↓
Model Registry
  ↓
Production Training
  ↓
Model Serving
  ↓
Testing
  ↓
Containerization
  ↓
Continuous Integration
```

The main takeaway is the separation between experimentation and production.

Experiments help answer:

> Which model should we use?

The production pipeline answers:

> How do we train, version, deploy, test, and reliably serve that model?