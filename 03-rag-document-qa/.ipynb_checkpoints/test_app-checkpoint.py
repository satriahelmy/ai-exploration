from dotenv import load_dotenv

from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService


load_dotenv()


embedding_client = EmbeddingClient()

vector_repository = VectorRepository()

retrieval_service = RetrievalService(
    embedding_client=embedding_client,
    vector_repository=vector_repository,
)

llm_client = LLMClient()

rag_service = RAGService(
    retrieval_service=retrieval_service,
    llm_client=llm_client,
)


question = "How many days of annual leave do employees receive?"

answer = rag_service.ask(question)

print(f"\nQuestion: {question}")
print(f"\nAnswer: {answer}")