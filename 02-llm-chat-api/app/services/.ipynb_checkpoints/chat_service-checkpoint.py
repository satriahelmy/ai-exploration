from app.clients.llm_client import LLMClient
from app.models import ChatMessage


class ChatService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def chat(
        self,
        message: str,
        history: list[ChatMessage],
    ) -> str:
        messages = [
            {
                "role": item.role,
                "content": item.content,
            }
            for item in history
        ]

        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        return self.llm_client.generate(messages)