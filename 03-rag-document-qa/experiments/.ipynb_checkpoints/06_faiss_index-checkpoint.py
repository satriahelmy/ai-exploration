import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Employees receive 18 working days of paid annual leave.",
    "The standard work week is 40 hours.",
    "Employees must return company laptops when leaving the company.",
]

query = "How many vacation days do employees get?"


# 1. Create document embeddings
document_embeddings = model.encode(documents)

print("Embedding shape:", document_embeddings.shape)


# 2. Normalize embeddings
document_embeddings = document_embeddings.astype("float32")
faiss.normalize_L2(document_embeddings)


# 3. Create FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)


# 4. Add document vectors to the index
index.add(document_embeddings)

print("Vectors in index:", index.ntotal)


# 5. Embed and normalize the query
query_embedding = model.encode([query]).astype("float32")

faiss.normalize_L2(query_embedding)


# 6. Search
TOP_K = 3

scores, indices = index.search(
    query_embedding,
    TOP_K,
)


# 7. Display results
print(f"\nQuery: {query}")

for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]),
    start=1,
):
    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Index: {index_id}")
    print(f"Text : {documents[index_id]}")