import mlflow

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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


samplers = {
    "random_oversampling": RandomOverSampler(random_state=42),
    "smote": SMOTE(random_state=42),
    "random_undersampling": RandomUnderSampler(random_state=42),
}


for sampler_name, sampler in samplers.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("sampler", sampler),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight=None,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    run_name = f"logreg_{sampler_name}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "C": 0.1,
                "class_weight": "none",
                "sampling_method": sampler_name,
                "threshold": 0.5,
            }
        )

        pipeline.fit(X_train, y_train)

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