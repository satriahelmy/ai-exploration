from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).parent.parent / "data"

COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(
            {
                "page": page_number,
                "text": page.extract_text(),
            }
        )

    return pages


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def create_chunks(pdf_path, document_id):
    pages = extract_pages(pdf_path)

    all_chunks = []

    for page in pages:
        chunks = chunk_text(
            page["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_index, text in enumerate(chunks):
            all_chunks.append(
                {
                    "document_id": document_id,
                    "filename": pdf_path.name,
                    "page": page["page"],
                    "chunk_id": chunk_index,
                    "text": text,
                }
            )

    return all_chunks


def ingest_document(pdf_path, model, client):
    if document_exists(client, pdf_path.name):
        print(f"\nSkipping existing document: {pdf_path.name}")
        return
    
    document_id = str(uuid4())

    chunks = create_chunks(
        pdf_path=pdf_path,
        document_id=document_id,
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload=chunk,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"\nIngested: {pdf_path.name}")
    print(f"Document ID: {document_id}")
    print(f"Chunks: {len(chunks)}")

def document_exists(client, filename):
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
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


model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    host="localhost",
    port=6333,
)


# Create collection only if it does not exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


pdf_files = list(DATA_DIR.glob("*.pdf"))

print(f"PDF files found: {len(pdf_files)}")


for pdf_path in pdf_files:
    ingest_document(
        pdf_path=pdf_path,
        model=model,
        client=client,
    )