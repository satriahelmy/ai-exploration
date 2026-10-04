from openai import OpenAI

from app.config import settings
from app.exceptions import LLMServiceError


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        self.model = settings.OPENAI_MODEL

    def generate(self, prompt):
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