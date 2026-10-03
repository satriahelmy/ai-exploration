from dotenv import load_dotenv

from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.models import AskRequest, AskResponse
from app.repositories.vector_repository import VectorRepository
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.ingestion_service import IngestionService


load_dotenv()


app = FastAPI(
    title="Multi-Document RAG API",
    version="0.1.0",
)

embedding_client = EmbeddingClient()

vector_repository = VectorRepository()

ingestion_service = IngestionService(
    embedding_client=embedding_client,
    vector_repository=vector_repository,
)

retrieval_service = RetrievalService(
    embedding_client=embedding_client,
    vector_repository=vector_repository,
)

llm_client = LLMClient()

rag_service = RAGService(
    retrieval_service=retrieval_service,
    llm_client=llm_client,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    answer = rag_service.ask(request.question)

    return AskResponse(
        answer=answer,
    )

@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    with NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        shutil.copyfileobj(
            file.file,
            temp_file,
        )

        temp_path = Path(temp_file.name)

    try:
        # Preserve the original filename for metadata
        original_name_path = temp_path.with_name(file.filename)
        temp_path.rename(original_name_path)

        result = ingestion_service.ingest(
            original_name_path
        )

        return result

    finally:
        if "original_name_path" in locals():
            original_name_path.unlink(
                missing_ok=True
            )

        temp_path.unlink(
            missing_ok=True
        )

@app.get("/documents")
def get_documents():
    return vector_repository.list_documents()

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    vector_repository.delete_document(
        document_id
    )

    return {
        "status": "deleted",
        "document_id": document_id,
    }