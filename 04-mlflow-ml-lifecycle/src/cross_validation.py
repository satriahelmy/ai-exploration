import mlflow
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from data import load_data, prepare_data, split_data
from preprocessing import build_preprocessor


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Load data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Combine train + validation for cross-validation
X_train_val = np.concatenate([X_train.index, X_val.index])
X_cv = X.loc[X_train_val]
y_cv = y.loc[X_train_val]


# Frozen candidate configuration
pipeline = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "model",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


with mlflow.start_run(run_name="logreg_5fold_cv"):

    results = cross_validate(
        pipeline,
        X_cv,
        y_cv,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    mlflow.log_params(
        {
            "model_type": "logistic_regression",
            "C": 0.1,
            "class_weight": "balanced",
            "cv_folds": 5,
        }
    )

    for metric in scoring:

        scores = results[f"test_{metric}"]

        mean_score = scores.mean()
        std_score = scores.std()

        mlflow.log_metric(f"cv_{metric}_mean", mean_score)
        mlflow.log_metric(f"cv_{metric}_std", std_score)

        print(f"\n{metric.upper()}")

        for fold, score in enumerate(scores, start=1):
            print(f"Fold {fold}: {score:.4f}")

        print(f"Mean: {mean_score:.4f}")
        print(f"Std : {std_score:.4f}")