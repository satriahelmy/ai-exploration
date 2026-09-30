from fastapi import FastAPI, HTTPException
from app.clients.llm_client import LLMClient
from app.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.exceptions import LLMServiceError
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
)

static_dir = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static",
)

llm_client = LLMClient()
chat_service = ChatService(llm_client)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        static_dir / "index.html"
    )


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = chat_service.chat(
            message=request.message,
            history=request.history,
        )

        return ChatResponse(answer=answer)

    except LLMServiceError:
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        )