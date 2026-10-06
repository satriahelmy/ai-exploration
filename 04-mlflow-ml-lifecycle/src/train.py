import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data import load_data, prepare_data, split_data
from preprocessing import build_preprocessor

import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
)


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Load and prepare data
df = load_data("data/telco_customer_churn.csv")

X, y = prepare_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Candidate models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}


# Train and evaluate each model
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        # Log model parameters to MLflow
        mlflow.log_param("model_type", model_name)

        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        pipeline.fit(X_train, y_train)

        # Train model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_prob)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

        # Create artifact directory
        artifact_dir = f"artifacts/{model_name}"
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
        
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        
        confusion_matrix_path = f"{artifact_dir}/confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # Classification report
        report = classification_report(y_val, y_pred)
        
        classification_report_path = f"{artifact_dir}/classification_report.txt"
        
        with open(classification_report_path, "w") as file:
            file.write(report)
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(
            artifact_dir,
            artifact_path="evaluation",
        )

        print(f"\nModel: {model_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")