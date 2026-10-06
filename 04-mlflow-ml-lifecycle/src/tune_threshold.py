import mlflow

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data import load_data, prepare_data, split_data
from preprocessing import build_preprocessor


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Load data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Best model configuration from previous experiment
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


# Train only once
pipeline.fit(X_train, y_train)

# Get probabilities on validation set
y_prob = pipeline.predict_proba(X_val)[:, 1]


thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
    }

    run_name = f"logreg_threshold_{threshold:.2f}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "C": 0.1,
                "class_weight": "balanced",
                "threshold": threshold,
            }
        )

        mlflow.log_metrics(metrics)

        print(f"\nRun: {run_name}")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")