import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model_name = "telco-churn-model"
model_version = "1"
alias = "champion"


client.set_registered_model_alias(
    name=model_name,
    alias=alias,
    version=model_version,
)


print(
    f"Alias '{alias}' assigned to "
    f"{model_name} version {model_version}"
)