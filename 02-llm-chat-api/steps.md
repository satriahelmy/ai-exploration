# Project 02 — LLM Chat API

## Goal

Learn how to build a proper software application around an external LLM API.

## Learning Flow

Browser UI
→ FastAPI
→ Chat Service
→ LLM Client
→ OpenAI API

---

## Progress

- [x] 01. First LLM API Call
- [x] 02. Messages, Roles & Context
- [x] 03. Build an LLM Client
- [x] 04. Build the Chat Service
- [x] 05. Add Pydantic Models
- [x] 06. Expose Chat with FastAPI
- [x] 07. Add Conversation History
- [x] 08. Configuration & Packaging
- [x] 09. Error Handling
- [x] 10. Testing & Mocking
- [x] 11. Docker
- [x] 12. GitHub Actions CI
- [x] 13. Simple Chat UI
- [x] 14. Final Documentation & Cleanup

---

## Step 01 — First LLM API Call

### Goal

Understand the smallest possible interaction with an LLM API.

### What We Learned

An application sends structured input to a remote model and receives a generated response.

### Key Takeaway

Calling an LLM API is easy. Engineering a reliable application around it is the larger problem.

---

## Step 02 — Messages, Roles & Context

### Goal

Understand how conversational context is represented.

### Concepts

- system messages
- user messages
- assistant messages
- conversation history

### Key Takeaway

LLMs do not automatically remember application conversations.
The application must provide the relevant context.

---

## Step 03 — Build an LLM Client

### Goal

Isolate provider-specific API communication.

### What We Built

A dedicated LLM client responsible for communicating with the OpenAI API.

### Key Takeaway

Provider-specific code should not be spread throughout the application.

---

## Step 04 — Build the Chat Service

### Goal

Separate application logic from external API communication.

### Responsibility

The chat service:

1. receives conversation history
2. adds the current user message
3. constructs the message sequence
4. calls the LLM client
5. returns the answer

### Key Takeaway

The service layer owns application behavior while the client layer owns external communication.

---

## Step 05 — Add Pydantic Models

### Goal

Define explicit API contracts.

### Main Models

- `ChatMessage`
- `ChatRequest`
- `ChatResponse`

### Key Takeaway

Structured contracts become increasingly important when LLM applications exchange complex data.

---

## Step 06 — Expose Chat with FastAPI

### Goal

Expose the chat service through an HTTP API.

### Architecture

HTTP Request
→ FastAPI
→ Chat Service
→ LLM Client
→ OpenAI

---

## Step 07 — Add Conversation History

### Goal

Support multi-turn conversations.

### Design Decision

The server remains stateless.

Conversation history is supplied by the client with each request.

### Key Takeaway

Conversation memory is an application architecture decision, not an automatic property of the LLM.

---

## Step 08 — Configuration & Packaging

### Goal

Separate configuration and secrets from source code.

### Configuration

Environment variables include:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

`.env` remains local while `.env.example` documents required configuration.

### Key Takeaway

Secrets should never be hardcoded or committed to Git.

---

## Step 09 — Error Handling

### Goal

Prevent provider failures from leaking directly through the application.

### What We Built

Provider errors are converted into an application-level error and ultimately
mapped to an appropriate HTTP response.

### Key Takeaway

External services fail. Applications need explicit failure boundaries.

---

## Step 10 — Testing & Mocking

### Goal

Test application behavior without making real LLM API calls.

### What We Learned

Mocking allows tests to be:

- deterministic
- fast
- inexpensive
- independent of external services

### Key Takeaway

Most application tests should not depend on a live LLM provider.

---

## Step 11 — Docker

### Goal

Run the LLM application inside a reproducible container.

### Important Design

Secrets are injected at runtime rather than baked into the image.

### Key Takeaway

Container images and runtime configuration should remain separate.

---

## Step 12 — GitHub Actions CI

### Goal

Automatically run tests for every relevant code change.

### Important Failure We Encountered

CI initially failed even though the LLM call was mocked.

The OpenAI client was being initialized during module import and expected an API key.

### Solution

Create the provider client lazily when it is actually needed.

### Key Takeaway

Mocking an API call is not enough if an external dependency is initialized before the test can replace it.

This was an important lesson about dependency lifecycle and testable architecture.

---

## Step 13 — Simple Chat UI

### Goal

Turn the backend API into a small usable application.

### What We Built

A lightweight browser interface using:

- HTML
- CSS
- JavaScript

The browser maintains conversation history and sends it to the backend.

### Final Architecture

Browser UI
→ FastAPI
→ Chat Service
→ LLM Client
→ OpenAI API

---

## Step 14 — Final Documentation & Cleanup

### Final Lesson

This project started with a simple LLM API call but gradually introduced the
engineering layers required to build a maintainable application.

The major lesson was:

> Calling an LLM is not the same as engineering an LLM application.