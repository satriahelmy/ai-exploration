from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Employees receive 18 working days of paid annual leave.",
    "The standard work week is 40 hours.",
    "Employees must return company laptops when leaving the company.",
]

embeddings = model.encode(texts)

print(f"Number of texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")

print("\nFirst text:")
print(texts[0])

print("\nFirst 10 embedding values:")
print(embeddings[0][:10])