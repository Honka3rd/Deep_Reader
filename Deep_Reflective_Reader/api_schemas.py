from pydantic import BaseModel, Field


class PrepareDocumentRequest(BaseModel):
    doc_name: str = Field(..., description="Document name")


class AskDocumentRequest(BaseModel):
    doc_name: str = Field(..., description="Document name")
    query: str = Field(..., description="User question")
    top_k: int = Field(3, ge=1, le=20)


class AskDocumentResponse(BaseModel):
    doc_name: str
    query: str
    answer: str


class StatusResponse(BaseModel):
    status: str
    message: str