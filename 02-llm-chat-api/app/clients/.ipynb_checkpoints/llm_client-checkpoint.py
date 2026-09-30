from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = "gpt-5.6"):
        self.client = OpenAI()
        self.model = model

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        return response.output_text