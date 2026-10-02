from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


COLLECTION_NAME = "rag_demo"

documents = [
    {
        "id": 0,
        "page": 2,
        "chunk_id": 4,
        "text": "Employees receive 18 working days of paid annual leave.",
    },
    {
        "id": 1,
        "page": 2,
        "chunk_id": 1,
        "text": "The standard work week is 40 hours.",
    },
    {
        "id": 2,
        "page": 6,
        "chunk_id": 1,
        "text": "Employees must return company laptops when leaving the company.",
    },
]


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to Qdrant
client = QdrantClient(
    host="localhost",
    port=6333,
)


# 3. Create embeddings
texts = [document["text"] for document in documents]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)


# 4. Recreate collection for this experiment
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE,
    ),
)


# 5. Insert vectors + payload
points = []

for document, embedding in zip(documents, embeddings):
    points.append(
        PointStruct(
            id=document["id"],
            vector=embedding.tolist(),
            payload={
                "page": document["page"],
                "chunk_id": document["chunk_id"],
                "text": document["text"],
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)


# 6. Search
query = "How many vacation days do employees get?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)


results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=3,
).points


# 7. Display results
print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):
    print("\n================================")
    print(f"Rank: {rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Point ID: {result.id}")
    print(f"Page: {result.payload['page']}")
    print(f"Chunk: {result.payload['chunk_id']}")
    print(f"Text: {result.payload['text']}")