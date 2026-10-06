import itertools

import mlflow

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


# Hyperparameter search space
c_values = [0.01, 0.1, 1.0, 10.0]
class_weights = [None, "balanced"]

combinations = itertools.product(
    c_values,
    class_weights,
)


for c_value, class_weight in combinations:

    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    weight_name = "none" if class_weight is None else class_weight
    run_name = f"logreg_c{c_value}_weight_{weight_name}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "C": c_value,
                "class_weight": weight_name,
            }
        )

        # Train only on training set
        pipeline.fit(X_train, y_train)

        # Evaluate only on validation set
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        mlflow.log_metrics(metrics)

        print(f"\nRun: {run_name}")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")