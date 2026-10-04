# AI Exploration

A hands-on repository documenting my journey in learning AI engineering by building projects from first principles.

Instead of focusing only on model training, these projects explore the engineering required to turn AI and machine learning models into usable applications and services.

## Why This Repository Exists

AI engineering involves more than training models.

It includes building APIs, managing dependencies, validating inputs, testing systems, containerizing applications, integrating external models, designing retrieval pipelines, working with tools and agents, and eventually deploying and observing AI systems in production.

This repository is my workspace for learning those concepts through progressively more complex projects.

## Projects

### 01 — Python AI API

A simple machine learning model served as a REST API.

**Topics explored:**

- model training and persistence
- inference services
- Python project structure
- Pydantic validation
- FastAPI
- automated testing with pytest
- Python packaging
- Docker
- GitHub Actions CI

Status: **Completed**

### 02 — LLM Chat API

An LLM-powered chat application built with FastAPI and the OpenAI API.

**Topics explored:**

- LLM API integration
- messages, roles, and conversation context
- LLM client abstraction
- service layer architecture
- Pydantic validation
- FastAPI
- multi-turn conversation history
- configuration and environment variables
- error handling
- automated testing and mocking
- Python packaging
- Docker
- GitHub Actions CI
- simple browser-based chat interface

Status: **Completed**

---

### 03 — Multi-Document RAG from Scratch

A multi-document Retrieval-Augmented Generation system built from first principles without using high-level RAG frameworks.

The project starts with the fundamentals of embeddings and semantic search, then progressively builds a complete document question-answering application using Qdrant, FastAPI, and OpenAI.

**Topics explored:**

- PDF text extraction and chunking
- embeddings with Sentence Transformers
- cosine similarity and semantic search
- vector search with FAISS
- vector databases with Qdrant
- retrieval pipelines
- grounded LLM generation
- source attribution
- multi-document retrieval
- document ingestion and management
- service-oriented RAG architecture
- FastAPI
- PDF upload
- automated testing and mocking
- retrieval evaluation with Hit@K and MRR
- error handling
- configuration and Python packaging
- Docker Compose
- GitHub Actions CI
- simple browser-based RAG interface

**Retrieval evaluation:**

- Hit@3: **100%**
- MRR: **0.875**

Status: **Completed**
