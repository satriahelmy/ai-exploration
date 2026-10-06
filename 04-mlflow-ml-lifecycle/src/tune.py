import itertools

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
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
learning_rates = [0.05, 0.1]
n_estimators_values = [100, 200]
max_depth_values = [2, 3]


combinations = itertools.product(
    learning_rates,
    n_estimators_values,
    max_depth_values,
)


for learning_rate, n_estimators, max_depth in combinations:

    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    run_name = (
        f"gb_lr{learning_rate}"
        f"_n{n_estimators}"
        f"_depth{max_depth}"
    )

    with mlflow.start_run(run_name=run_name):

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "gradient_boosting",
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            }
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        # Evaluate
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