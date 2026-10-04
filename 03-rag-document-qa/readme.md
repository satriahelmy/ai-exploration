# Multi-Document RAG from Scratch

A multi-document Retrieval-Augmented Generation (RAG) system built from scratch using FastAPI, Sentence Transformers, Qdrant, and OpenAI.

The project demonstrates the complete RAG pipeline, from document ingestion and vector retrieval to grounded answer generation, evaluation, testing, containerization, and CI.

The implementation intentionally avoids high-level RAG frameworks such as LangChain in order to make the underlying retrieval and generation pipeline explicit.

---

## Overview

Large Language Models cannot reliably answer questions about private or domain-specific documents that are not available in their training data.

Retrieval-Augmented Generation solves this by retrieving relevant information from an external knowledge base and providing that information to the LLM as context.

This project implements that process end-to-end:

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
LLM
      ↓
Grounded Answer + Sources
```

The final application supports multiple PDF documents and provides a simple web interface for document management and question answering.

---

## Features

- Multi-PDF document ingestion
- PDF text extraction
- Overlapping text chunking
- Sentence Transformer embeddings
- Semantic vector search
- Qdrant vector database
- Cross-document retrieval
- Grounded LLM generation
- Document and page source attribution
- Duplicate document detection
- Document listing and deletion
- FastAPI REST API
- PDF upload endpoint
- RAG service architecture
- Retrieval evaluation
- Unit testing with mocks
- Application-level error handling
- Environment-based configuration
- Docker Compose deployment
- GitHub Actions CI
- Simple web interface

---

## Architecture

```text
                        ┌─────────────────────┐
                        │      Browser        │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │       FastAPI       │
                        └──────────┬──────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
       Document Ingestion                        RAG Service
                │                                     │
                ▼                                     ▼
          PDF Extraction                       Retrieval Service
                │                                     │
                ▼                                     ▼
             Chunking                         Query Embedding
                │                                     │
                ▼                                     ▼
           Embeddings                              Qdrant
                │                                     │
                └──────────────► Qdrant ◄─────────────┘
                                                      │
                                                      ▼
                                             Relevant Chunks
                                                      │
                                                      ▼
                                                 LLM Client
                                                      │
                                                      ▼
                                                   OpenAI
                                                      │
                                                      ▼
                                           Grounded Answer
                                             + Source Citation
```

---

## RAG Pipeline

### 1. Document Extraction

PDF documents are processed using `pypdf`.

Text is extracted page by page so that page metadata can be preserved for source attribution.

Each extracted page contains:

```text
page
text
```

---

### 2. Chunking

Document text is divided into overlapping chunks.

Default configuration:

```text
Chunk size:    500 characters
Chunk overlap: 100 characters
```

Each chunk stores metadata such as:

```text
document_id
filename
page
chunk_id
text
```

The overlap helps preserve context that may otherwise be split across chunk boundaries.

---

### 3. Embeddings

Chunks are converted into dense vector representations using:

```text
all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

The same embedding model is used for both document chunks and user questions.

This allows document chunks and queries to be compared within the same vector space.

---

### 4. Vector Search

The project first explores vector similarity manually using cosine similarity and FAISS.

The final application uses Qdrant as the vector database.

Qdrant stores:

```text
Vector
+
Payload
```

The payload contains the original chunk text and its metadata.

The collection uses:

```text
Vector size: 384
Distance:    Cosine
```

---

### 5. Semantic Retrieval

When a user submits a question:

```text
Question
   ↓
Sentence Transformer
   ↓
Query Embedding
   ↓
Qdrant Search
   ↓
Top-K Relevant Chunks
```

The vector itself is not converted back into text.

Instead, Qdrant identifies the nearest stored vectors and returns their associated payloads containing the original document text.

---

### 6. Grounded Generation

Retrieved chunks are combined into a context supplied to the LLM.

The generation prompt instructs the model to:

- answer only using the provided context
- avoid unsupported information
- return an explicit unknown response when the context is insufficient
- cite the supporting document and page
- cite only sources that support the answer

Example:

```text
Question:
How many annual leave days do employees receive?

Answer:
Full-time employees receive 18 working days of paid annual leave per calendar year.
(sample_employee_handbook.pdf, Page 2)
```

If the answer is not supported by the retrieved context:

```text
I don't know based on the provided context.
```

---

## Multi-Document Retrieval

All document chunks are stored in the same Qdrant collection.

This allows a question to retrieve relevant information across multiple documents:

```text
Document A ─┐
Document B ─┼─► Qdrant
Document C ─┘
                ↑
             Question
```

The application does not require the user to manually select a document before asking a question.

---

## Document Management

The application supports:

```text
GET    /documents
POST   /documents
DELETE /documents/{document_id}
```

Documents receive unique IDs during ingestion.

Duplicate filenames are detected before ingestion.

Deleting a document removes its associated vectors from Qdrant.

---

## Project Structure

```text
03-rag-document-qa/
│
├── app/
│   ├── clients/
│   │   ├── embedding_client.py
│   │   └── llm_client.py
│   │
│   ├── repositories/
│   │   └── vector_repository.py
│   │
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_service.py
│   │
│   ├── static/
│   │   ├── app.js
│   │   └── style.css
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── config.py
│   ├── exceptions.py
│   ├── main.py
│   └── models.py
│
├── evaluation/
│   ├── retrieval_dataset.json
│   └── evaluate_retrieval.py
│
├── tests/
│   └── test_rag_service.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Service Architecture

Responsibilities are separated into dedicated layers.

### `EmbeddingClient`

Responsible for converting text into vector embeddings using Sentence Transformers.

### `VectorRepository`

Handles Qdrant operations:

- collection initialization
- vector insertion
- semantic search
- duplicate detection
- document listing
- document deletion

### `IngestionService`

Handles the document ingestion pipeline:

```text
PDF
→ Extraction
→ Chunking
→ Embedding
→ Vector Storage
```

### `RetrievalService`

Handles:

```text
Question
→ Query Embedding
→ Vector Search
→ Relevant Chunks
```

### `RAGService`

Orchestrates:

```text
Retrieval
→ Context Construction
→ Grounded Prompt
→ LLM Generation
```

### `LLMClient`

Provides the abstraction between the application and the LLM provider.

---

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

### Ask a Question

```http
POST /ask
```

Request:

```json
{
  "question": "How many annual leave days do employees receive?"
}
```

Example response:

```json
{
  "answer": "Full-time employees receive 18 working days of paid annual leave per calendar year. (sample_employee_handbook.pdf, Page 2)"
}
```

---

### Upload Document

```http
POST /documents
```

The endpoint accepts a PDF file using multipart form data.

The ingestion pipeline automatically:

```text
Extracts text
→ Creates chunks
→ Generates embeddings
→ Stores vectors in Qdrant
```

---

### List Documents

```http
GET /documents
```

Returns documents currently available in the vector database.

---

### Delete Document

```http
DELETE /documents/{document_id}
```

Deletes all vectors associated with the specified document.

---

## Retrieval Evaluation

Retrieval quality is evaluated independently from LLM generation.

The evaluation dataset contains questions with expected source documents and pages.

Metrics:

### Hit@K

Measures whether the expected source appears anywhere within the top K retrieved results.

### Mean Reciprocal Rank

Measures how highly the first relevant result is ranked.

```text
MRR = mean(1 / rank of first relevant result)
```

### Current Results

```text
Questions: 4
Hits:      4
Hit@3:     100.00%
MRR:       0.875
```

Results by query:

```text
Annual leave       → Relevant source at rank 1
Sick leave         → Relevant source at rank 1
Lost laptop        → Relevant source at rank 1
Equipment return   → Relevant source at rank 2
```

These metrics evaluate the retrieval component separately from generation quality.

---

## Testing

Unit tests focus on the orchestration logic of the RAG service.

External dependencies are mocked so tests do not require:

- a running Qdrant instance
- OpenAI API access
- network connectivity

Run tests with:

```bash
python -m pytest
```

Current test result:

```text
2 passed
```

---

## Error Handling

The application defines domain-specific exceptions:

```text
RAGError
├── VectorStoreError
├── LLMServiceError
└── DocumentProcessingError
```

Examples:

```text
Invalid PDF
→ HTTP 400

Qdrant unavailable
→ HTTP 503

LLM provider unavailable
→ HTTP 503
```

This prevents infrastructure exceptions from leaking directly through the API.

---

## Configuration

Configuration is loaded from environment variables.

Create a `.env` file based on:

```text
.env.example
```

Example:

```env
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-5.6

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_documents

EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

The real `.env` file is excluded from Git.

---

## Running Locally

### 1. Install Dependencies

```bash
pip install ".[dev]"
```

### 2. Start Qdrant

A local Qdrant instance must be available at the configured host and port.

### 3. Start the API

```bash
python -m uvicorn app.main:app --reload --port 8001
```

### 4. Open the Application

Web interface:

```text
http://127.0.0.1:8001/
```

API documentation:

```text
http://127.0.0.1:8001/docs
```

---

## Docker Compose

The complete application can be started using Docker Compose.

```bash
docker compose up --build
```

The Compose stack contains:

```text
Docker Compose
├── rag-api
│   ├── FastAPI
│   ├── Sentence Transformer
│   └── RAG Application
│
└── qdrant
    └── Vector Database
```

Qdrant data is stored in a Docker volume so vectors persist across normal container restarts.

The application container communicates with Qdrant through the internal Compose network:

```text
rag-api
   ↓
qdrant:6333
```

The application automatically initializes the required Qdrant collection when necessary.

---

## Continuous Integration

GitHub Actions runs automated tests when relevant Project 03 files change.

CI pipeline:

```text
Push / Pull Request
        ↓
GitHub Actions
        ↓
Python 3.11
        ↓
Install Dependencies
        ↓
Run Unit Tests
        ↓
Pass / Fail
```

The workflow is scoped to the Project 03 directory because this project lives inside a monorepo.

---

## Web Interface

The project includes a minimal web interface built with:

- HTML
- CSS
- Vanilla JavaScript

The UI supports:

- viewing available documents
- uploading PDFs
- deleting documents
- asking questions
- displaying grounded answers

The frontend communicates directly with the FastAPI endpoints.

No frontend framework is required.

---

## Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Validation | Pydantic |
| PDF Processing | pypdf |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | Qdrant |
| LLM | OpenAI API |
| Testing | pytest |
| Containerization | Docker |
| Multi-Service Runtime | Docker Compose |
| CI | GitHub Actions |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## Key Concepts Learned

This project was built incrementally to understand the RAG stack rather than treating RAG as a black box.

### Retrieval Fundamentals

- PDF text extraction
- text chunking
- embeddings
- cosine similarity
- semantic search
- FAISS
- approximate nearest-neighbor search concepts

### RAG

- retrieval pipelines
- context construction
- grounded prompting
- source attribution
- multi-document retrieval

### Vector Databases

- vector storage
- payload metadata
- cosine distance
- Qdrant collections
- vector search
- document-level deletion

### Application Engineering

- service architecture
- FastAPI
- configuration management
- domain-specific error handling
- unit testing
- mocking
- Python packaging

### RAG Quality

- golden retrieval datasets
- Hit@K
- Mean Reciprocal Rank
- retrieval evaluation independent from generation

### Production Foundations

- Docker
- Docker Compose
- persistent vector storage
- GitHub Actions
- CI
- environment-based configuration

---

## Development Progression

The project intentionally progresses from low-level retrieval concepts toward a complete application:

```text
PDF Extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
Cosine Similarity
      ↓
Semantic Search
      ↓
FAISS
      ↓
Retrieval Pipeline
      ↓
RAG Generation
      ↓
Grounded Prompting
      ↓
Source Attribution
      ↓
Qdrant
      ↓
Multi-Document Retrieval
      ↓
Document Management
      ↓
Service Architecture
      ↓
FastAPI
      ↓
Testing
      ↓
Evaluation
      ↓
Error Handling
      ↓
Packaging
      ↓
Docker Compose
      ↓
CI
      ↓
Web UI
```

---

## What This Project Intentionally Does Not Include

To keep the project focused on RAG fundamentals, the following are intentionally excluded:

- LangChain
- AI agents
- reranking
- hybrid search
- knowledge graphs
- authentication
- OCR
- complex frontend frameworks

These capabilities belong to later projects where they can be introduced with a clear understanding of the underlying system.

---

## Main Takeaway

RAG is not simply:

```text
Upload PDF → Ask LLM
```

A production-oriented RAG system consists of several independent engineering problems:

```text
Document Processing
        +
Embedding
        +
Vector Retrieval
        +
Context Construction
        +
Grounded Generation
        +
Evaluation
        +
Application Engineering
        +
Infrastructure
```

Building each layer explicitly makes it possible to understand where a RAG system succeeds, where it fails, and which component needs to be improved.