import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "rag_documents",
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2",
    )

    CHUNK_SIZE = int(
        os.getenv("CHUNK_SIZE", "500")
    )

    CHUNK_OVERLAP = int(
        os.getenv("CHUNK_OVERLAP", "100")
    )


settings = Settings()