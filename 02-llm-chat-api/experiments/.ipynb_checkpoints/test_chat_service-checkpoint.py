from dotenv import load_dotenv

from app.clients.llm_client import LLMClient
from app.services.chat_service import ChatService


load_dotenv()

llm_client = LLMClient()

chat_service = ChatService(llm_client)

answer = chat_service.chat(
    "Explain overfitting in one sentence."
)

print(answer)