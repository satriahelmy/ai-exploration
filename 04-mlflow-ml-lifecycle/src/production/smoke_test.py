from src.data import load_data, prepare_data, split_data
from src.production.load_model import (
    load_production_model,
    predict_churn,
)


# Prepare sample data
df = load_data("data/telco_customer_churn.csv")
X, y = prepare_data(df)

_, _, X_test, _, _, _ = split_data(X, y)

sample = X_test.head(5)


# Load champion model from MLflow Registry
model = load_production_model()


# Run inference
predictions, probabilities = predict_churn(
    model,
    sample,
)


for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities),
    start=1,
):
    print(
        f"Customer {i}: "
        f"churn_probability={probability:.4f}, "
        f"prediction={prediction}"
    )