import os

from openai import OpenAI

from app.exceptions import LLMServiceError
from app.exceptions import VectorStoreError

class LLMClient:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6")
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )

            return response.output_text

        except Exception as exc:
            raise LLMServiceError(
                "Failed to generate response from LLM."
            ) from exc