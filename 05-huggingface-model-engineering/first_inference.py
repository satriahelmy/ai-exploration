
import torch
from transformers import pipeline

MODEL_ID = "dslim/bert-base-NER"

device = 0 if torch.cuda.is_available() else -1

print("Model:", MODEL_ID)
print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")

ner_pipeline = pipeline(
    task="token-classification",
    model=MODEL_ID,
    aggregation_strategy="simple",
    device=device,
)

text = "Satya Nadella works at Microsoft in Seattle."

entities = ner_pipeline(text)

for entity in entities:
    print(
        f"Entity: {entity['word']:<20} "
        f"Type: {entity['entity_group']:<5} "
        f"Score: {entity['score']:.4f}"
    )
