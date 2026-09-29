from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify.",
    )


class PredictionResponse(BaseModel):
    """Output schema for sentiment prediction."""

    label: str
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )