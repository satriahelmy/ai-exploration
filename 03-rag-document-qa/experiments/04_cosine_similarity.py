from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Employees receive 18 working days of paid annual leave.",
    "The standard work week is 40 hours.",
    "Employees must return company laptops when leaving the company.",
]

query = "How many vacation days do employees get?"


document_embeddings = model.encode(documents)
query_embedding = model.encode([query])


similarities = cosine_similarity(
    query_embedding,
    document_embeddings,
)[0]


print(f"Query: {query}\n")

for document, score in zip(documents, similarities):
    print(f"Score: {score:.4f}")
    print(f"Text : {document}")
    print()