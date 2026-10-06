# AI Exploration

A project-based journey to learn AI Engineering by building real systems from scratch.

The goal of this repository is not only to learn how AI models work, but also how to turn them into usable, testable, and production-oriented applications.

Each project introduces a new layer of the AI engineering stack.

## Why This Repository?

AI Engineering covers more than model training.

A production AI system may involve:

- data and preprocessing
- machine learning models
- LLMs and embeddings
- APIs and inference services
- vector databases
- experiment tracking
- model registries
- testing and CI
- containerization
- deployment

Instead of learning these topics separately, this repository explores them through progressively more complete projects.

## Projects

### 01 — Python AI API ✅

Build a traditional machine learning model and expose it as a production-style API.

**Key concepts:**

- TF-IDF
- Logistic Regression
- model training and persistence
- inference service
- Pydantic
- FastAPI
- testing with pytest
- Python packaging
- Docker
- GitHub Actions CI

**Main takeaway:**

A machine learning model becomes much more useful when it is packaged as a reliable software service.

---

### 02 — LLM Chat API ✅

Build a chat application using an LLM API while separating application logic from the model provider.

**Key concepts:**

- LLM APIs
- system prompts
- chat roles
- conversation history
- service architecture
- environment variables
- error handling
- FastAPI
- testing and mocking
- Docker
- CI
- simple web interface

**Main takeaway:**

An LLM application is more than an API call. It needs application state, interfaces, error handling, testing, and a clean service architecture.

---

### 03 — Multi-Document RAG from Scratch ✅

Build a Retrieval-Augmented Generation system that answers questions using information retrieved from multiple PDF documents.

**Architecture:**

```text id="p03arch"
PDF Documents
     ↓
Text Extraction
     ↓
Chunking
     ↓
Embeddings
     ↓
Vector Store
     ↓
Semantic Retrieval
     ↓
Context Construction
     ↓
LLM
     ↓
Grounded Answer + Sources
```

**Key concepts:**

- PDF text extraction
- chunking
- sentence embeddings
- cosine similarity
- FAISS
- Qdrant
- semantic retrieval
- metadata and payloads
- grounded prompting
- source citations
- retrieval evaluation
- FastAPI
- web interface
- Docker Compose
- CI

**Main takeaway:**

RAG does not teach an LLM new knowledge. It retrieves relevant information and provides that information as context when generating an answer.

---

### 04 — MLflow & ML Lifecycle ✅

Build an end-to-end machine learning lifecycle around a customer churn prediction system.

**Architecture:**

```text id="p04arch"
Dataset
   ↓
Data Preparation
   ↓
Model Experiments
   ↓
MLflow Tracking
   ↓
Model Evaluation
   ↓
Model Registry
   ↓
@champion
   ↓
FastAPI
   ↓
Docker Compose
   ↓
CI
```

**Key concepts:**

- train / validation / test strategy
- baseline modeling
- model comparison
- hyperparameter experiments
- class imbalance handling
- threshold tuning
- cross-validation
- MLflow experiment tracking
- parameters, metrics, and artifacts
- model logging
- MLflow Model Registry
- model versions and aliases
- reproducible production training
- FastAPI inference
- API testing and mocking
- Docker Compose
- GitHub Actions CI

**Final model:**

```text id="p04model"
Logistic Regression
C = 0.1
class_weight = balanced
prediction threshold = 0.55
```

**Final test performance:**

| Metric | Score |
|---|---:|
| Accuracy | 0.7559 |
| Precision | 0.5285 |
| Recall | 0.7433 |
| F1 | 0.6178 |
| ROC-AUC | 0.8414 |

**Main takeaway:**

Machine learning engineering does not end at `model.fit()`.

A model needs an engineering lifecycle around it so experiments can be tracked, models can be evaluated and versioned, and the selected model can be reliably served, tested, and deployed.

## Learning Progression

```text id="progression"
Machine Learning Model
        ↓
Production-style API
        ↓
LLM Application
        ↓
Retrieval-Augmented Generation
        ↓
ML Lifecycle & Model Management
```

More projects will be added as the exploration continues.