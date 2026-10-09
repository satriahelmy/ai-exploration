
from transformers import AutoTokenizer

MODEL_ID = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = "Satya Nadella works at Microsoft in Seattle."

tokens = tokenizer.tokenize(text)

print("Original text:", text)
print("\nTokens:")

for index, token in enumerate(tokens):
    print(f"{index:2d}: {token}")

print("\nToken IDs:")
print(tokenizer.convert_tokens_to_ids(tokens))
