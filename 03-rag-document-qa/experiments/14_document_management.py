from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


COLLECTION_NAME = "rag_documents"

client = QdrantClient(
    host="localhost",
    port=6333,
)


def list_documents():
    documents = {}

    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload

            document_id = payload["document_id"]

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": payload["filename"],
                    "chunks": 0,
                }

            documents[document_id]["chunks"] += 1

        if offset is None:
            break

    return list(documents.values())

def document_exists(filename):
    documents = list_documents()

    return any(
        document["filename"] == filename
        for document in documents
    )

def delete_document(document_id):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    ),
                )
            ]
        ),
    )

documents = list_documents()

print("\nDocuments:")

for document in documents:
    print("\n------------------------------")
    print(f"Document ID: {document['document_id']}")
    print(f"Filename: {document['filename']}")
    print(f"Chunks: {document['chunks']}")

print(
    "\nEmployee handbook exists:",
    document_exists("sample_employee_handbook.pdf"),
)

print(
    "Unknown document exists:",
    document_exists("unknown.pdf"),
)

