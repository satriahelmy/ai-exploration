from dotenv import load_dotenv

from app.clients.llm_client import LLMClient


load_dotenv()

llm = LLMClient()

messages = [
    {
        "role": "user",
        "content": "Explain machine learning in one sentence.",
    }
]

answer = llm.generate(messages)

print(answer)