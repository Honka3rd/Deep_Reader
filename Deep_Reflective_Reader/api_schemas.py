from pydantic import BaseModel, Field, model_validator


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
    task_unit_split_mode: str | None = Field(
        None,
        description=(
            "Task-unit split mode: semantic_safe | progressive | llm_enhanced. "
            "This controls task-unit resolution only, not structured parser mode."
        ),
    )
    semantic_top_k_candidates: int | None = Field(
        None,
        description=(
            "Optional semantic rerank top-k for semantic_safe split mode. "
            "Larger values may improve semantic cut precision but can be slower."
        ),
    )
    refresh_summary: bool = Field(
        False,
        description=(
            "When true, force regenerate section/chapter summary and overwrite cache. "
            "When false, reuse cached summary artifact when valid."
        ),
    )
    refresh_quiz: bool = Field(
        False,
        description=(
            "When true, force regenerate section/chapter quiz and overwrite cache. "
            "When false, reuse cached quiz artifact when valid."
        ),
    )


class SectionTaskResponse(BaseModel):
    """Response payload for section task endpoints."""

    doc_name: str
    section_id: str
    success: bool
    result: str | None
    reason: str | None
    cache_hit: bool | None = None


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
    cache_hit: bool | None = None


class SummarizeChapterRequest(BaseModel):
    """Request payload for chapter-summary endpoint."""

    doc_name: str = Field(..., description="Document name")
    chapter_title: str | None = Field(
        None,
        description=(
            "Exact chapter title. Optional when chapter_id is provided. "
            "When both chapter_id and chapter_title are provided, chapter_id wins."
        ),
    )
    chapter_id: str | None = Field(
        None,
        description=(
            "Stable chapter identifier from task-layout response. "
            "Preferred target for duplicate chapter titles."
        ),
    )
    task_unit_split_mode: str | None = Field(
        None,
        description=(
            "Task-unit split mode: semantic_safe | progressive | llm_enhanced. "
            "This controls task-unit resolution only, not structured parser mode."
        ),
    )
    semantic_top_k_candidates: int | None = Field(
        None,
        description=(
            "Optional semantic rerank top-k for semantic_safe split mode. "
            "Larger values may improve semantic cut precision but can be slower."
        ),
    )
    refresh_summary: bool = Field(
        False,
        description=(
            "When true, force regenerate chapter summary and overwrite cache. "
            "When false, reuse cached chapter summary artifact when valid."
        ),
    )

    @model_validator(mode="after")
    def _validate_target(self) -> "SummarizeChapterRequest":
        chapter_id = (self.chapter_id or "").strip()
        chapter_title = (self.chapter_title or "").strip()
        if not chapter_id and not chapter_title:
            raise ValueError("chapter_id or chapter_title must be provided")
        if not chapter_id:
            self.chapter_id = None
        else:
            self.chapter_id = chapter_id
        if not chapter_title:
            self.chapter_title = None
        else:
            self.chapter_title = chapter_title
        return self


class SummarizeChapterResponse(BaseModel):
    """Response payload for chapter-summary endpoint."""

    doc_name: str
    chapter_title: str
    success: bool
    result: str | None
    reason: str | None
    cache_hit: bool | None = None


class ChapterQuizRequest(BaseModel):
    """Request payload for chapter-quiz endpoint."""

    doc_name: str = Field(..., description="Document name")
    chapter_title: str | None = Field(
        None,
        description=(
            "Exact chapter title. Optional when chapter_id is provided. "
            "When both chapter_id and chapter_title are provided, chapter_id wins."
        ),
    )
    chapter_id: str | None = Field(
        None,
        description=(
            "Stable chapter identifier from task-layout response. "
            "Preferred target for duplicate chapter titles."
        ),
    )
    task_unit_split_mode: str | None = Field(
        None,
        description=(
            "Task-unit split mode: semantic_safe | progressive | llm_enhanced. "
            "This controls task-unit resolution only, not structured parser mode."
        ),
    )
    semantic_top_k_candidates: int | None = Field(
        None,
        description=(
            "Optional semantic rerank top-k for semantic_safe split mode. "
            "Larger values may improve semantic cut precision but can be slower."
        ),
    )
    refresh_quiz: bool = Field(
        False,
        description=(
            "When true, force regenerate chapter quiz and overwrite cache. "
            "When false, reuse cached chapter quiz artifact when valid."
        ),
    )

    @model_validator(mode="after")
    def _validate_target(self) -> "ChapterQuizRequest":
        chapter_id = (self.chapter_id or "").strip()
        chapter_title = (self.chapter_title or "").strip()
        if not chapter_id and not chapter_title:
            raise ValueError("chapter_id or chapter_title must be provided")
        if not chapter_id:
            self.chapter_id = None
        else:
            self.chapter_id = chapter_id
        if not chapter_title:
            self.chapter_title = None
        else:
            self.chapter_title = chapter_title
        return self


class ChapterQuizResponse(BaseModel):
    """Response payload for chapter-quiz endpoint."""

    doc_name: str
    chapter_title: str
    success: bool
    questions: list[QuizQuestionResponse] | None
    reason: str | None
    cache_hit: bool | None = None


class ArtifactAvailabilityResponse(BaseModel):
    """Lightweight artifact availability metadata without heavy task payload."""

    has_summary: bool = False
    has_quiz: bool = False
    summary_cache_valid: bool | None = None
    quiz_cache_valid: bool | None = None
    summary_invalid_reason: str | None = None
    quiz_invalid_reason: str | None = None
    summary_generated_at: str | None = None
    quiz_generated_at: str | None = None


class GetDocumentTaskLayoutRequest(BaseModel):
    """Request payload for document task-layout endpoint."""

    doc_name: str = Field(..., description="Document name")
    refresh_task_units: bool = Field(
        False,
        description=(
            "When true, force recompute task units and overwrite persisted task-unit cache. "
            "When false, reuse persisted section.task_units when cache is valid."
        ),
    )
    task_unit_split_mode: str | None = Field(
        None,
        description=(
            "Task-unit split mode: semantic_safe | progressive | llm_enhanced. "
            "This controls task-unit resolution only, not structured parser mode."
        ),
    )
    semantic_top_k_candidates: int | None = Field(
        None,
        description=(
            "Optional semantic rerank top-k for semantic_safe split mode. "
            "Larger values may improve semantic cut precision but can be slower."
        ),
    )


class TaskUnitMetadataResponse(BaseModel):
    """Task-unit metadata payload for frontend layout rendering."""

    unit_id: str
    title: str | None
    container_title: str | None
    source_section_ids: list[str]
    is_fallback_generated: bool
    artifacts: ArtifactAvailabilityResponse | None = None


class SectionTaskLayoutResponse(BaseModel):
    """Section layout response node with embedded task-unit metadata."""

    section_id: str
    title: str | None
    container_title: str | None
    section_role: str | None
    parent_chapter_id: str | None
    section_kind: str | None
    is_implicit_section: bool = False
    task_mode: str
    task_units: list[TaskUnitMetadataResponse]
    artifacts: ArtifactAvailabilityResponse | None = None


class DocumentTaskLayoutChapterResponse(BaseModel):
    """Hierarchy-first chapter response node."""

    chapter_id: str
    title: str | None
    level: int
    chapter_role: str | None
    sections: list[SectionTaskLayoutResponse]
    artifacts: ArtifactAvailabilityResponse | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EnhancedParseRecommendationResponse(BaseModel):
    """Enhanced parser recommendation payload for layout consumers."""

    should_recommend: bool
    score: int
    reasons: list[str]
    metrics: dict[str, float | int]


class ProfileStructureDiagnosticsResponse(BaseModel):
    """Lightweight diagnostics for target-safety and observability.

    Mixed-source semantics:
    - shape/risk hints are profile-derived snapshot signals;
    - task-unit availability/coverage reflects current task-layout state.
    """

    parser_metadata_shape: str | None = None
    post_actual_structure_shape: str | None = None
    title_uniqueness_risk: str | None = None
    title_target_requires_id: bool = False
    task_unit_stats_available: bool = False
    task_unit_section_coverage: float | None = None
    parser_post_shape_mismatch: bool = False
    enhanced_parse_hint: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DocumentTaskLayoutResponse(BaseModel):
    """Response payload for reading current effective task-layout snapshot."""

    document_id: str
    title: str
    language: str | None
    chapters: list[DocumentTaskLayoutChapterResponse]
    enhanced_parse_recommendation: EnhancedParseRecommendationResponse | None
    profile_diagnostics: ProfileStructureDiagnosticsResponse | None = None


class ReparseDocumentStructureRequest(BaseModel):
    """Request payload for explicit structure reparse action."""

    doc_name: str = Field(..., description="Document name")
    parser_mode: str = Field(
        ...,
        description="Parser mode: common | llm_enhanced",
    )


class ReparseDocumentStructureResponse(BaseModel):
    """Response payload for structure reparse action."""

    success: bool
    doc_name: str
    parser_mode: str
    structured_document_path: str | None
    error: str | None
    section_count: int | None
