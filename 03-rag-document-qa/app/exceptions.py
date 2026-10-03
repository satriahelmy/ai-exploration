class RAGError(Exception):
    """Base exception for RAG application errors."""


class VectorStoreError(RAGError):
    """Raised when the vector database operation fails."""


class LLMServiceError(RAGError):
    """Raised when the LLM provider operation fails."""


class DocumentProcessingError(RAGError):
    """Raised when document processing fails."""