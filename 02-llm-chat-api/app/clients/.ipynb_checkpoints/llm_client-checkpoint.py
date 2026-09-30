from openai import OpenAI, OpenAIError

from app.config import settings
from app.exceptions import LLMServiceError


class LLMClient:
    def __init__(self):
        self.model = settings.openai_model

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not settings.openai_api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is not configured."
            )

        try:
            client = OpenAI(
                api_key=settings.openai_api_key
            )

            response = client.responses.create(
                model=self.model,
                input=messages,
            )

            return response.output_text

        except OpenAIError as error:
            raise LLMServiceError(
                "The LLM provider failed to process the request."
            ) from error