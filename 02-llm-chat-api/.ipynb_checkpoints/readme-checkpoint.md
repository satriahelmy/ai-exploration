# LLM Chat API

A production-style LLM chat application built with FastAPI and the OpenAI API.

## Features

- LLM-powered chat
- Multi-turn conversation history
- FastAPI REST API
- Pydantic request validation
- Configuration via environment variables
- LLM provider error handling
- Automated tests with mocking
- Docker support
- GitHub Actions CI
- Simple browser-based chat interface

## Architecture

Browser UI
    ↓
FastAPI
    ↓
Chat Service
    ↓
LLM Client
    ↓
OpenAI API

## Project Structure

...

## Setup

...

## Running the Application

...

## API

### GET /health

### POST /chat

...

## Testing

...

## Docker

...

## CI

...

## What I Learned

- Separating API, service, and provider layers
- Managing conversation context
- Designing testable LLM applications
- Mocking external LLM calls
- Handling provider failures
- Managing secrets and configuration
- Containerizing an LLM application
- Building CI without exposing API credentials