from uuid import uuid4

from pypdf import PdfReader
from app.exceptions import DocumentProcessingError

from app.config import settings


class IngestionService:
    def __init__(
        self,
        embedding_client,
        vector_repository,
        chunk_size=500,
        chunk_overlap=100,
    ):
        self.embedding_client = embedding_client
        self.vector_repository = vector_repository
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pages(self, file_path):
        try:
            reader = PdfReader(file_path)
    
            pages = []
    
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
    
                if text:
                    pages.append(
                        {
                            "page": page_number,
                            "text": text,
                        }
                    )
    
            if not pages:
                raise DocumentProcessingError(
                    "No extractable text found in the PDF."
                )
    
            return pages
    
        except DocumentProcessingError:
            raise
    
        except Exception as exc:
            raise DocumentProcessingError(
                "Failed to process the PDF document."
            ) from exc

    def chunk_text(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunks.append(text[start:end])

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def ingest(self, pdf_path):
        filename = pdf_path.name

        if self.vector_repository.document_exists(filename):
            return {
                "status": "skipped",
                "filename": filename,
            }

        document_id = str(uuid4())

        pages = self.extract_pages(pdf_path)

        chunks = []

        for page in pages:
            page_chunks = self.chunk_text(page["text"])

            for chunk_id, text in enumerate(page_chunks):
                chunks.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page": page["page"],
                        "chunk_id": chunk_id,
                        "text": text,
                    }
                )

        embeddings = self.embedding_client.embed(
            [chunk["text"] for chunk in chunks]
        )

        self.vector_repository.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "status": "ingested",
            "document_id": document_id,
            "filename": filename,
            "chunks": len(chunks),
        }