from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    status: str
    document_id: str | None = None
    filename: str
    chunks: int | None = None