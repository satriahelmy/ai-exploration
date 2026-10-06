import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data import load_data, prepare_data, split_data
from preprocessing import build_preprocessor


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn")


# Enable scikit-learn autologging
mlflow.sklearn.autolog()


# Load and prepare data
df = load_data("data/telco_customer_churn.csv")

X, y = prepare_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# Build pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


with mlflow.start_run(run_name="logistic_regression_autolog"):

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_val, y_val)

    print(f"Validation accuracy: {score:.4f}")