import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data import load_data, prepare_data, split_data
from preprocessing import build_preprocessor


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Load data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Combine train + validation for production training
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])


# Frozen model configuration
model = LogisticRegression(
    C=0.1,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ]
)


# Train production candidate
pipeline.fit(X_train_val, y_train_val)


with mlflow.start_run(run_name="production_candidate_logistic_regression") as run:

    mlflow.log_params(
        {
            "model_type": "logistic_regression",
            "C": 0.1,
            "class_weight": "balanced",
            "threshold": 0.55,
            "training_data": "train+validation",
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