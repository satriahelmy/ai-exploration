from pydantic import ValidationError

from app.models import ChatRequest


valid_request = ChatRequest(
    message="What is machine learning?"
)

print("Valid request:")
print(valid_request)

print()

try:
    invalid_request = ChatRequest(
        message=""
    )
except ValidationError as error:
    print("Validation error:")
    print(error)