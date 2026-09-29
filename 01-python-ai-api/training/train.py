from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# =========================================================
# Training data
# =========================================================

texts = [
    "the application is very useful",
    "I really like this application",
    "the features are very helpful",
    "I had a great experience using this application",
    "the application works perfectly",
    "I am very satisfied with this service",
    "the application is excellent",
    "this app makes my work much easier",
    "the service is fast and reliable",
    "I would recommend this application",

    "the application is terrible",
    "I am disappointed with this application",
    "the features are useless",
    "the application keeps crashing",
    "I had a terrible experience using this application",
    "I am very dissatisfied with this service",
    "the application is very frustrating",
    "this app is difficult to use",
    "the service is slow and unreliable",
    "I would not recommend this application",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",

    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]


# =========================================================
# Build model pipeline
# =========================================================

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
            ),
        ),
    ]
)


# =========================================================
# Train model
# =========================================================

model.fit(texts, labels)


# =========================================================
# Test model
# =========================================================

sample_text = ["this application is really helpful"]

prediction = model.predict(sample_text)[0]
probabilities = model.predict_proba(sample_text)[0]

print(f"Sample text : {sample_text[0]}")
print(f"Prediction  : {prediction}")

for label, probability in zip(model.classes_, probabilities):
    print(f"{label:<10}: {probability:.4f}")


# =========================================================
# Save model artifact
# =========================================================

project_root = Path(__file__).resolve().parent.parent
artifact_dir = project_root / "artifacts"

artifact_dir.mkdir(parents=True, exist_ok=True)

model_path = artifact_dir / "sentiment_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")