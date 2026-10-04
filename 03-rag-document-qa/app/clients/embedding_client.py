from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingClient:
    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

    def embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def embed_query(self, query):
        return self.embed([query])[0]