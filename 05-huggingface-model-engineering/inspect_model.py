from transformers import AutoConfig

MODEL_ID = "dslim/bert-base-NER"

config = AutoConfig.from_pretrained(MODEL_ID)

print("Model type:", config.model_type)
print("Hidden size:", config.hidden_size)
print("Number of layers:", config.num_hidden_layers)
print("Attention heads:", config.num_attention_heads)
print("Number of labels:", config.num_labels)
print("Labels:", config.id2label)