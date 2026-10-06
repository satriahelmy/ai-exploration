import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")


run_id = "a7e9fdc6e04e41978825cf352bffde7b"
model_uri = f"runs:/{run_id}/model"

model_name = "telco-churn-model"


result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)


print(f"Model name: {result.name}")
print(f"Model version: {result.version}")
print(f"Status: {result.status}")