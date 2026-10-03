from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)


class VectorRepository:
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="rag_documents",
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
        )
        self.collection_name = collection_name

    def add_chunks(self, chunks, embeddings):
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding.tolist(),
                    payload=chunk,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_embedding, limit=5):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
        ).points

    def document_exists(self, filename):
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return len(points) > 0

    def delete_document(self, document_id):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    def list_documents(self):
        documents = {}
        offset = None
    
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
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