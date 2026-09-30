from fastapi import FastAPI, HTTPException
from app.clients.llm_client import LLMClient
from app.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.exceptions import LLMServiceError

app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
)

llm_client = LLMClient()
chat_service = ChatService(llm_client)


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