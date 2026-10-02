from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


# 1. Extract PDF
pages = extract_pages(PDF_PATH)


# 2. Create chunks
all_chunks = []

for page in pages:
    chunks = chunk_text(
        page["text"],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    for chunk_index, chunk in enumerate(chunks):
        all_chunks.append(
            {
                "page": page["page"],
                "chunk_id": chunk_index,
                "text": chunk,
            }
        )


print(f"Total chunks: {len(all_chunks)}")


# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Embed all chunks
chunk_texts = [chunk["text"] for chunk in all_chunks]

chunk_embeddings = model.encode(chunk_texts)


# 5. User query
query = "How many vacation days do employees receive?"

query_embedding = model.encode([query])


# 6. Compare query with every chunk
similarities = cosine_similarity(
    query_embedding,
    chunk_embeddings,
)[0]


# 7. Rank chunks from highest to lowest similarity
ranked_indices = similarities.argsort()[::-1]


# 8. Take the top K results
top_indices = ranked_indices[:TOP_K]


print(f"\nQuery: {query}")
print(f"\nTop {TOP_K} results:")


for rank, index in enumerate(top_indices, start=1):
    chunk = all_chunks[index]
    score = similarities[index]

    print("\n================================")
    print(f"Rank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk_id']}")
    print("--------------------------------")
    print(chunk["text"])