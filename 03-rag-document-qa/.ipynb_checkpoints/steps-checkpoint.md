# Project 03 — Multi-Document RAG from Scratch

## Goal

Learn how Retrieval-Augmented Generation works from first principles before
using high-level RAG frameworks.

## Target Architecture

Multiple PDFs
→ Text Extraction
→ Chunking
→ Embeddings
→ Vector Search
→ Retrieval
→ Relevant Context
→ LLM
→ Grounded Answer + Sources

---

# Milestone A — Retrieval Fundamentals

- [x] 01. PDF Text Extraction
- [ ] 02. Document Chunking
- [ ] 03. Embeddings
- [ ] 04. Cosine Similarity
- [ ] 05. Semantic Search

# Milestone B — Basic RAG

- [ ] 06. FAISS Vector Index
- [ ] 07. Retrieval Pipeline
- [ ] 08. RAG Generation
- [ ] 09. Grounded Prompting
- [ ] 10. Source Attribution

# Milestone C — Vector Database

- [ ] 11. Qdrant

# Milestone D — Multi-Document

- [ ] 12. Multi-PDF Ingestion
- [ ] 13. Cross-Document Retrieval
- [ ] 14. Document Management

# Milestone E — Application Engineering

- [ ] 15. RAG Service Architecture
- [ ] 16. FastAPI
- [ ] 17. PDF Upload

# Milestone F — Quality

- [ ] 18. Testing & Mocking
- [ ] 19. RAG Evaluation
- [ ] 20. Error Handling
- [ ] 21. Configuration & Packaging

# Milestone G — Shipping

- [ ] 22. Docker Compose
- [ ] 23. GitHub Actions CI
- [ ] 24. Simple Web UI
- [ ] 25. Final Documentation

---

## Step 01 — PDF Text Extraction

### Goal

Convert a PDF document into text that can be processed by Python while
preserving page information.

### What We Built

Used `pypdf` to:

1. open a text-based PDF
2. iterate through its pages
3. extract text from each page
4. preserve the page number

### Current Representation

```python
{
    "page": 1,
    "text": "..."
}