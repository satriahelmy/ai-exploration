import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_data, prepare_data, split_data
from src.preprocessing import build_preprocessor
from src.production.config import (
    RANDOM_STATE,
    MODEL_C,
    MODEL_CLASS_WEIGHT,
    MODEL_MAX_ITER,
    PREDICTION_THRESHOLD,
)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Load and prepare data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Final training data: train + validation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])


# Frozen production configuration
model = LogisticRegression(
    C=MODEL_C,
    class_weight=MODEL_CLASS_WEIGHT,
    max_iter=MODEL_MAX_ITER,
    random_state=RANDOM_STATE,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ]
)


# Train final model
pipeline.fit(X_train_val, y_train_val)


# Log production candidate
with mlflow.start_run(run_name="production_logistic_regression") as run:

    mlflow.log_params(
        {
            "model_type": "logistic_regression",
            "C": MODEL_C,
            "class_weight": MODEL_CLASS_WEIGHT,
            "max_iter": MODEL_MAX_ITER,
            "random_state": RANDOM_STATE,
            "threshold": PREDICTION_THRESHOLD,
            "training_data": "train+validation",
            "training_rows": len(X_train_val),
            "sklearn_version": sklearn.__version__,
        }
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Training rows: {len(X_train_val)}")
    print(f"Threshold: {PREDICTION_THRESHOLD}")