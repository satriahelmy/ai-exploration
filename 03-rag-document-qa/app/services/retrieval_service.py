class RetrievalService:
    def __init__(
        self,
        embedding_client,
        vector_repository,
    ):
        self.embedding_client = embedding_client
        self.vector_repository = vector_repository

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_client.embed_query(query)

        results = self.vector_repository.search(
            query_embedding=query_embedding,
            limit=top_k,
        )

        return results