from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(__file__).parent.parent / "data" / "sample_employee_handbook.pdf"


reader = PdfReader(PDF_PATH)

print(f"Number of pages: {len(reader.pages)}")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- PAGE {page_number} ---")
    print(text[:500])