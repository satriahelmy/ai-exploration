from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input="Explain what an API is in one short paragraph."
)

print(response.output_text)