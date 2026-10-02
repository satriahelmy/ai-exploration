from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "rag_documents"
TOP_K = 5


model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    host="localhost",
    port=6333,
)


def retrieve(query, top_k=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
    ).points

    return results


query = "When must company equipment be returned?"

results = retrieve(
    query=query,
    top_k=TOP_K,
)


print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):
    payload = result.payload

    print("\n================================")
    print(f"Rank: {rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Document: {payload['filename']}")
    print(f"Document ID: {payload['document_id']}")
    print(f"Page: {payload['page']}")
    print(f"Chunk: {payload['chunk_id']}")
    print("--------------------------------")
    print(payload["text"])