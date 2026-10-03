import os

from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL")

    def generate(self, prompt):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text