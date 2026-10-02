from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_PATH = Path(__file__).parent.parent / "data" / "sample_employee_handbook.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


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


def create_chunks(pages):
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
                    "page": page["page"],
                    "chunk_id": chunk_index,
                    "text": text,
                }
            )

    return all_chunks


def build_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        chunk = chunks[index_id]

        results.append(
            {
                "score": float(score),
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
        )

    return results


# ----------------------------------
# Build retrieval system
# ----------------------------------

pages = extract_pages(PDF_PATH)

chunks = create_chunks(pages)

model = SentenceTransformer("all-MiniLM-L6-v2")

index = build_index(chunks, model)


print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"Vectors in index: {index.ntotal}")


# ----------------------------------
# Query
# ----------------------------------

query = "How many vacation days do employees receive?"

results = retrieve(
    query=query,
    model=model,
    index=index,
    chunks=chunks,
    top_k=TOP_K,
)


print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):
    print("\n================================")
    print(f"Rank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Page: {result['page']}")
    print(f"Chunk: {result['chunk_id']}")
    print("--------------------------------")
    print(result["text"])