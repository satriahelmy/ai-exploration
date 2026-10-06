import mlflow
import mlflow.sklearn

from data import load_data, prepare_data, split_data


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# MLflow run containing the logged model
# run_id = "a7e9fdc6e04e41978825cf352bffde7b"

model_uri = "models:/telco-churn-model@champion"


# Load trained pipeline from MLflow
model = mlflow.sklearn.load_model(model_uri)


# Try inference on a few samples
sample = X_test.head(5)

probabilities = model.predict_proba(sample)[:, 1]

threshold = 0.55
predictions = (probabilities >= threshold).astype(int)


for i, (probability, prediction) in enumerate(
    zip(probabilities, predictions),
    start=1,
):
    print(
        f"Customer {i}: "
        f"churn_probability={probability:.4f}, "
        f"prediction={prediction}"
    )