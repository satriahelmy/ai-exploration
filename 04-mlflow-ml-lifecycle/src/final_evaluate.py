import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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


# Frozen configuration selected using validation set
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


# Train on training set
pipeline.fit(X_train, y_train)


# Final evaluation on test set
y_prob = pipeline.predict_proba(X_test)[:, 1]
threshold = 0.55
y_pred = (y_prob >= threshold).astype(int)


metrics = {
    "test_accuracy": accuracy_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_recall": recall_score(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, y_prob),
}


with mlflow.start_run(run_name="final_logistic_regression"):

    mlflow.log_params(
        {
            "model_type": "logistic_regression",
            "C": 0.1,
            "class_weight": "balanced",
            "threshold": threshold,
        }
    )

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
    )

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")