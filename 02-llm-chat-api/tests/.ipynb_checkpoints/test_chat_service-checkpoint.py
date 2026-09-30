from app.models import ChatMessage
from app.services.chat_service import ChatService


class FakeLLMClient:
    def __init__(self):
        self.received_messages = None

    def generate(self, messages):
        self.received_messages = messages
        return "Mocked LLM response."


def test_chat_service_builds_messages():
    fake_llm = FakeLLMClient()
    service = ChatService(fake_llm)

    history = [
        ChatMessage(
            role="user",
            content="My name is Helmy.",
        ),
        ChatMessage(
            role="assistant",
            content="Nice to meet you!",
        ),
    ]

    answer = service.chat(
        message="What is my name?",
        history=history,
    )

    assert answer == "Mocked LLM response."

    assert fake_llm.received_messages == [
        {
            "role": "user",
            "content": "My name is Helmy.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you!",
        },
        {
            "role": "user",
            "content": "What is my name?",
        },
    ]