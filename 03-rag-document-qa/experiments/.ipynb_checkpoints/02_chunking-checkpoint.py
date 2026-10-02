from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(__file__).parent.parent / "data" / "sample_employee_handbook.pdf"

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

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


pages = extract_pages(PDF_PATH)

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


print(f"Pages: {len(pages)}")
print(f"Chunks: {len(all_chunks)}")

for chunk in all_chunks[:5]:
    print("\n------------------------------")
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk_id']}")
    print("------------------------------")
    print(chunk["text"])