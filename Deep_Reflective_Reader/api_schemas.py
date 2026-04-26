from pydantic import BaseModel, Field


class PrepareDocumentRequest(BaseModel):
    """Request payload for document preparation operations."""

    doc_name: str = Field(..., description="Document name")
    mode: str = Field("base", description="Preparation mode: base | free_qa")
    force_rebuild: bool = Field(
        False,
        description="When true, force artifact rebuild for selected mode.",
    )
    structured_parser_mode: str = Field(
        "common",
        description="Structured parser mode: common | llm_enhanced",
    )


class PrepareDocumentResponse(BaseModel):
    """Response payload for document preparation operations."""

    doc_name: str
    mode: str
    structured_parser_mode: str
    success: bool
    structured_document_ready: bool
    structured_document_path: str | None
    faiss_ready: bool
    profile_ready: bool
    bundle_ready: bool
    errors: list[str]


class AskDocumentRequest(BaseModel):
    """Request payload for document QA endpoint."""
    doc_name: str = Field(..., description="Document name")
    query: str = Field(..., description="User question")
    top_k: int = Field(3, ge=1, le=20)
    session_id: str | None = Field(None, description="Reading session id")


class AskDocumentResponse(BaseModel):
    """Response payload for document QA endpoint."""
    doc_name: str
    query: str
    answer: str
    session_id: str


class StatusResponse(BaseModel):
    """Response payload for API health/status endpoint."""
    status: str
    message: str


class SectionTaskRequest(BaseModel):
    """Request payload for section-summary / section-quiz endpoints."""

    doc_name: str = Field(..., description="Document name")
    section_id: str = Field(..., description="Target structured section id")


class SectionTaskResponse(BaseModel):
    """Response payload for section task endpoints."""

    doc_name: str
    section_id: str
    success: bool
    result: str | None
    reason: str | None


class QuizQuestionResponse(BaseModel):
    """Structured quiz question response DTO."""

    question_id: str
    question_text: str
    answer_text: str


class SectionQuizResponse(BaseModel):
    """Response payload for section-quiz endpoint."""

    doc_name: str
    section_id: str
    success: bool
    questions: list[QuizQuestionResponse] | None
    reason: str | None


class SummarizeChapterRequest(BaseModel):
    """Request payload for chapter-summary endpoint."""

    doc_name: str = Field(..., description="Document name")
    chapter_title: str = Field(..., description="Exact chapter title")


class SummarizeChapterResponse(BaseModel):
    """Response payload for chapter-summary endpoint."""

    doc_name: str
    chapter_title: str
    success: bool
    result: str | None
    reason: str | None
