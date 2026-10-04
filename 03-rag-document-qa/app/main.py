from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.exceptions import (
    DocumentProcessingError,
    LLMServiceError,
    VectorStoreError,
)
from app.models import AskRequest, AskResponse
from app.repositories.vector_repository import VectorRepository
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.config import settings

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

load_dotenv()


app = FastAPI(
    title="Multi-Document RAG API",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)

templates = Jinja2Templates(
    directory="app/templates"
)


# Dependencies
embedding_client = EmbeddingClient()

vector_repository = VectorRepository()

ingestion_service = IngestionService(
    embedding_client=embedding_client,
    vector_repository=vector_repository,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
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

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        answer = rag_service.ask(
            request.question
        )

        return AskResponse(
            answer=answer
        )

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except LLMServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post("/documents")
def upload_document(
    file: UploadFile = File(...)
):
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
        original_name_path = temp_path.with_name(
            file.filename
        )

        temp_path.rename(
            original_name_path
        )

        result = ingestion_service.ingest(
            original_name_path
        )

        return result

    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

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
    try:
        return vector_repository.list_documents()

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str
):
    try:
        vector_repository.delete_document(
            document_id
        )

        return {
            "status": "deleted",
            "document_id": document_id,
        }

    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc