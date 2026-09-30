from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input=[
        {
            "role": "user",
            "content": "My name is Helmy."
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Helmy!"
        },
        {
            "role": "user",
            "content": "What is my name?"
        }
    ]
)

print(response.output_text)