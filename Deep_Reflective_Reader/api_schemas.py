from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator
from shared.artifact_target_model import ArtifactTargetLevel

ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS: frozenset[str] = frozenset(
    {
        "source_hash",
        "content_block_id",
        "quote_span_start",
        "quote_span_end",
        "schema_version",
    }
)


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


class PrepareTaskLayoutRequest(BaseModel):
    """Request payload for prepare-if-needed task-layout orchestration."""

    doc_name: str = Field(..., description="Document name")
    force_rebuild: bool = Field(
        False,
        description="When true, rebuild preparation artifacts before loading layout.",
    )
    structured_parser_mode: str = Field(
        "common",
        description="Structured parser mode: common | llm_enhanced",
    )
    refresh_task_units: bool = Field(
        False,
        description="When true, recompute task units for the returned layout.",
    )
    task_unit_split_mode: str | None = Field(
        None,
        description=(
            "Task-unit split mode: semantic_safe | progressive | llm_enhanced. "
            "This controls task-unit resolution only."
        ),
    )
    semantic_top_k_candidates: int | None = Field(
        None,
        description="Optional semantic rerank top-k for semantic_safe task splitting.",
    )
    include_anchor_page_evidence: bool = Field(
        False,
        description=(
            "When true, explicitly load lightweight page-boundary evidence for "
            "TOC edit-existing anchor prefill. Ordinary reader calls should keep "
            "the default false value to avoid preparation/OCR source-evidence work."
        ),
    )


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


class DocumentListItemResponse(BaseModel):
    """Lightweight document candidate for document list/search API."""

    doc_name: str
    title: str | None = None
    source: str


class DocumentListResponse(BaseModel):
    """Response payload for lightweight document discovery/search."""

    items: list[DocumentListItemResponse]
    query: str | None = None
    total: int


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
    include_anchor_page_evidence: bool = Field(
        False,
        description=(
            "When true, explicitly load lightweight page-boundary evidence for "
            "TOC edit-existing anchor prefill. Ordinary reader calls should keep "
            "the default false value to avoid preparation/OCR source-evidence work."
        ),
    )


class GetTaskUnitContentRequest(BaseModel):
    """Path-parameter request contract for on-demand task-unit content lookup."""

    doc_name: str = Field(..., description="Document name")
    task_unit_id: str = Field(
        ...,
        description="Stable task unit id from task-layout response",
    )
    segmented: bool = Field(
        False,
        description=(
            "When true, return explicit opt-in deterministic segmented content blocks. "
            "When false, preserve compatibility-safe default content-block behavior."
        ),
    )
    include_raw_content: bool = Field(
        False,
        description=(
            "When true, include legacy raw task-unit `content` string for compatibility/debug. "
            "When false, prefer content-block-first payload without raw content duplication."
        ),
    )


class BatchTaskUnitContentRequest(BaseModel):
    """Request payload for batched on-demand task-unit content lookup."""

    task_unit_ids: list[str] = Field(
        ...,
        description="Ordered stable task-unit ids from task-layout response.",
    )
    segmented: bool = Field(
        False,
        description=(
            "When true, return explicit opt-in deterministic segmented content blocks "
            "for every requested task unit."
        ),
    )
    include_raw_content: bool = Field(
        False,
        description=(
            "When true, include legacy raw task-unit `content` strings for compatibility/debug. "
            "When false, prefer content-block-first payload without raw content duplication."
        ),
    )

    @model_validator(mode="after")
    def _validate_task_unit_ids(self) -> "BatchTaskUnitContentRequest":
        normalized_ids: list[str] = []
        seen: set[str] = set()
        duplicates: list[str] = []
        for task_unit_id in self.task_unit_ids:
            normalized_task_unit_id = task_unit_id.strip()
            if not normalized_task_unit_id:
                raise ValueError("task_unit_ids cannot contain empty values")
            if normalized_task_unit_id in seen:
                duplicates.append(normalized_task_unit_id)
            seen.add(normalized_task_unit_id)
            normalized_ids.append(normalized_task_unit_id)

        if not normalized_ids:
            raise ValueError("task_unit_ids cannot be empty")
        if duplicates:
            raise ValueError(
                "duplicate task_unit_ids in request: "
                + ", ".join(sorted(set(duplicates)))
            )

        self.task_unit_ids = normalized_ids
        return self


class TaskUnitMetadataResponse(BaseModel):
    """Task-unit metadata payload for frontend layout rendering."""

    unit_id: str
    title: str | None
    container_title: str | None
    source_section_ids: list[str]
    is_fallback_generated: bool
    artifacts: ArtifactAvailabilityResponse | None = None


class ArtifactTargetRefResponse(BaseModel):
    """Public API metadata-only artifact target reference."""

    _ALLOWED_METADATA_KEYS: ClassVar[frozenset[str]] = ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS

    target_level: ArtifactTargetLevel
    document_id: str | None = None
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    content_block_id: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_content_endpoint_target_constraints(self) -> "ArtifactTargetRefResponse":
        if self.target_level == ArtifactTargetLevel.CONTENT_BLOCK:
            if not (self.task_unit_id or "").strip():
                raise ValueError(
                    "content_block target_level requires task_unit_id in task-unit content response"
                )
            if not (self.content_block_id or "").strip():
                raise ValueError(
                    "content_block target_level requires content_block_id in task-unit content response"
                )

        if self.target_level == ArtifactTargetLevel.TASK_UNIT:
            if not (self.task_unit_id or "").strip():
                raise ValueError(
                    "task_unit target_level requires task_unit_id in task-unit content response"
                )

        if self.metadata is not None:
            unknown_keys = set(self.metadata.keys()) - self._ALLOWED_METADATA_KEYS
            if unknown_keys:
                raise ValueError(
                    "artifact target metadata contains unsupported keys: "
                    + ", ".join(sorted(unknown_keys))
                )
        return self


class TaskUnitContentBlockResponse(BaseModel):
    """Official API content-block contract for task-unit on-demand response."""

    block_id: str
    content: str
    block_type: str | None = None
    artifact_ids: list[str] | None = None
    artifact_target_refs: list[ArtifactTargetRefResponse] | None = None
    metadata: dict[str, Any] | None = None


class TaskUnitContentResponse(BaseModel):
    """On-demand task-unit content response with backward-compatible rich-content shape.

    Compatibility contract:
    - `content_blocks` is the primary render/interaction payload.
    - `content` is compatibility/debug-only and may be null by default.
    """

    document_id: str
    document_title: str
    task_unit_id: str
    title: str | None
    container_title: str | None
    content: str | None
    content_blocks: list[TaskUnitContentBlockResponse]
    source_section_ids: list[str]
    parent_section_id: str | None
    section_id: str | None
    section_title: str | None
    chapter_id: str | None
    chapter_title: str | None
    is_fallback_generated: bool


class BatchTaskUnitContentResponse(BaseModel):
    """Batched on-demand task-unit content response preserving request order."""

    document_id: str
    document_title: str
    contents: list[TaskUnitContentResponse]


class AnchorEvidenceResponse(BaseModel):
    """Lightweight existing-structure anchor evidence for task-layout prefill."""

    anchor_type: str | None
    status: str
    reason: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    page_start_index: int | None = None
    page_end_index: int | None = None
    page_start_label: str | None = None
    page_end_label: str | None = None


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
    anchor_evidence: AnchorEvidenceResponse | None = None


class DocumentTaskLayoutChapterResponse(BaseModel):
    """Hierarchy-first chapter response node."""

    chapter_id: str
    title: str | None
    level: int
    chapter_role: str | None
    sections: list[SectionTaskLayoutResponse]
    artifacts: ArtifactAvailabilityResponse | None = None
    anchor_evidence: AnchorEvidenceResponse | None = None
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


class ParseProvenanceResponse(BaseModel):
    """Lightweight parser provenance for task-layout observability."""

    requested_parser_mode: str | None = Field(
        None,
        description="Parser mode requested for the accepted structured hierarchy.",
    )
    effective_parser_mode: str | None = Field(
        None,
        description="Parser mode that actually produced the accepted hierarchy.",
    )
    fallback_used: bool = Field(
        False,
        description="True when requested parsing fell back before acceptance.",
    )
    fallback_reason: str | None = Field(
        None,
        description="Stable reason code when parser fallback was used.",
    )
    source: str | None = Field(
        None,
        description="Internal provenance source for debugging only.",
    )


class DocumentTaskLayoutResponse(BaseModel):
    """Response payload for reading current effective task-layout snapshot."""

    document_id: str
    title: str
    language: str | None
    chapters: list[DocumentTaskLayoutChapterResponse]
    enhanced_parse_recommendation: EnhancedParseRecommendationResponse | None
    profile_diagnostics: ProfileStructureDiagnosticsResponse | None = None
    parse_provenance: ParseProvenanceResponse | None = None


class ManualStructureAnchorRequest(BaseModel):
    """Typed source-agnostic anchor for user-supplied structure entries."""

    anchor_type: str = Field(
        ...,
        description="Manual structure anchor type: char_range | page_range",
    )
    char_start: int | None = Field(
        None,
        ge=0,
        description="Inclusive raw-text character start offset for char_range anchors.",
    )
    char_end: int | None = Field(
        None,
        ge=0,
        description="Exclusive raw-text character end offset for char_range anchors.",
    )
    page_start_index: int | None = Field(
        None,
        ge=0,
        description="Zero-based source page start index for page_range anchors.",
    )
    page_end_index: int | None = Field(
        None,
        ge=0,
        description="Optional zero-based source page end index for page_range anchors.",
    )

    @model_validator(mode="after")
    def _validate_anchor_shape(self) -> "ManualStructureAnchorRequest":
        normalized_anchor_type = self.anchor_type.strip().lower().replace("-", "_")
        if normalized_anchor_type not in {"char_range", "page_range"}:
            raise ValueError("anchor_type must be char_range or page_range")
        self.anchor_type = normalized_anchor_type

        has_char_fields = self.char_start is not None or self.char_end is not None
        has_page_fields = (
            self.page_start_index is not None or self.page_end_index is not None
        )
        if has_char_fields and has_page_fields:
            raise ValueError("manual structure anchor must not mix char and page fields")

        if normalized_anchor_type == "char_range":
            if self.char_start is None:
                raise ValueError("char_range anchor requires char_start")
            if self.page_start_index is not None or self.page_end_index is not None:
                raise ValueError("char_range anchor cannot include page fields")
            if self.char_end is not None and self.char_end <= self.char_start:
                raise ValueError("char_end must be greater than char_start")
            return self

        if self.page_start_index is None:
            raise ValueError("page_range anchor requires page_start_index")
        if self.char_start is not None or self.char_end is not None:
            raise ValueError("page_range anchor cannot include char fields")
        if self.page_end_index is not None and self.page_end_index < self.page_start_index:
            raise ValueError("page_end_index must be greater than or equal to page_start_index")
        return self


class ManualStructureEntryRequest(BaseModel):
    """User-supplied chapter/section boundary entry for manual structure validation."""

    title: str = Field(..., description="User-supplied chapter or section title.")
    level: int = Field(
        ...,
        ge=1,
        le=2,
        description="Manual structure level: 1=chapter, 2=section.",
    )
    anchor: ManualStructureAnchorRequest
    external_id: str | None = Field(
        None,
        description="Optional user/client id for correlating validation errors.",
    )
    notes: str | None = Field(
        None,
        description="Optional user-facing notes; not parser authority.",
    )

    @model_validator(mode="after")
    def _normalize_entry(self) -> "ManualStructureEntryRequest":
        normalized_title = self.title.strip()
        if not normalized_title:
            raise ValueError("manual structure entry title cannot be empty")
        self.title = normalized_title

        if self.external_id is not None:
            normalized_external_id = self.external_id.strip()
            self.external_id = normalized_external_id or None
        if self.notes is not None:
            normalized_notes = self.notes.strip()
            self.notes = normalized_notes or None
        return self


class ManualStructurePlanRequest(BaseModel):
    """Source-agnostic user-supplied structure plan for validation/preview."""

    entries: list[ManualStructureEntryRequest] = Field(
        ...,
        min_length=1,
        description="Ordered user-supplied chapter/section entries.",
    )
    source_hash: str | None = Field(
        None,
        description="Optional client-observed source hash for stale-source checks.",
    )

    @model_validator(mode="after")
    def _validate_entry_order(self) -> "ManualStructurePlanRequest":
        has_chapter = False
        for entry in self.entries:
            if entry.level == 1:
                has_chapter = True
                continue
            if not has_chapter:
                raise ValueError("manual structure section entry requires preceding chapter")

        if self.source_hash is not None:
            normalized_source_hash = self.source_hash.strip()
            self.source_hash = normalized_source_hash or None
        return self


class ManualStructureValidationRequest(BaseModel):
    """Request payload for manual structure validation/preview."""

    doc_name: str = Field(..., description="Document name")
    manual_structure: ManualStructurePlanRequest

    @model_validator(mode="after")
    def _normalize_doc_name(self) -> "ManualStructureValidationRequest":
        normalized_doc_name = self.doc_name.strip()
        if not normalized_doc_name:
            raise ValueError("doc_name cannot be empty")
        self.doc_name = normalized_doc_name
        return self


class ManualStructureValidationIssueResponse(BaseModel):
    """Validation/preview issue for a user-supplied manual structure plan."""

    _ALLOWED_CODES: ClassVar[frozenset[str]] = frozenset(
        {
            "malformed_payload",
            "unsupported_anchor_type",
            "out_of_range_anchor",
            "overlapping_range",
            "empty_projected_range",
            "invalid_level_sequence",
            "unsupported_depth",
            "stale_source_evidence",
        }
    )
    _ALLOWED_SEVERITIES: ClassVar[frozenset[str]] = frozenset(
        {"error", "warning", "info"}
    )

    code: str = Field(
        ...,
        description="Stable manual structure validation issue code.",
    )
    message: str = Field(
        ...,
        description="Human-readable validation or preview issue message.",
    )
    severity: str = Field(
        "error",
        description="Issue severity: error | warning | info.",
    )
    entry_index: int | None = Field(
        None,
        ge=0,
        description="Zero-based index into manual_structure.entries when applicable.",
    )
    external_id: str | None = Field(
        None,
        description="Optional user/client id copied from the related manual entry.",
    )

    @model_validator(mode="after")
    def _normalize_issue(self) -> "ManualStructureValidationIssueResponse":
        normalized_code = self.code.strip().lower().replace("-", "_")
        if normalized_code not in self._ALLOWED_CODES:
            raise ValueError("unsupported manual structure validation issue code")
        self.code = normalized_code

        normalized_severity = self.severity.strip().lower()
        if normalized_severity not in self._ALLOWED_SEVERITIES:
            raise ValueError("manual structure validation issue severity must be error, warning, or info")
        self.severity = normalized_severity

        normalized_message = self.message.strip()
        if not normalized_message:
            raise ValueError("manual structure validation issue message cannot be empty")
        self.message = normalized_message

        if self.external_id is not None:
            normalized_external_id = self.external_id.strip()
            self.external_id = normalized_external_id or None
        return self


class ManualStructureNormalizedEntryResponse(BaseModel):
    """Normalized manual structure entry returned by validation/preview."""

    title: str
    level: int = Field(
        ...,
        ge=1,
        le=2,
        description="Manual structure level: 1=chapter, 2=section.",
    )
    anchor: ManualStructureAnchorRequest
    external_id: str | None = None
    projected_char_start: int | None = Field(None, ge=0)
    projected_char_end: int | None = Field(None, ge=0)
    projected_page_start_index: int | None = Field(None, ge=0)
    projected_page_end_index: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _normalize_response_entry(self) -> "ManualStructureNormalizedEntryResponse":
        normalized_title = self.title.strip()
        if not normalized_title:
            raise ValueError("manual structure normalized entry title cannot be empty")
        self.title = normalized_title

        if self.external_id is not None:
            normalized_external_id = self.external_id.strip()
            self.external_id = normalized_external_id or None

        if (
            self.projected_char_start is not None
            and self.projected_char_end is not None
            and self.projected_char_end <= self.projected_char_start
        ):
            raise ValueError("projected_char_end must be greater than projected_char_start")
        if (
            self.projected_page_start_index is not None
            and self.projected_page_end_index is not None
            and self.projected_page_end_index < self.projected_page_start_index
        ):
            raise ValueError(
                "projected_page_end_index must be greater than or equal to projected_page_start_index"
            )
        return self


class ManualStructurePreviewSectionResponse(BaseModel):
    """Lightweight projected section node for manual structure preview."""

    title: str
    external_id: str | None = None
    entry_index: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _normalize_preview_section(self) -> "ManualStructurePreviewSectionResponse":
        normalized_title = self.title.strip()
        if not normalized_title:
            raise ValueError("manual structure preview section title cannot be empty")
        self.title = normalized_title
        if self.external_id is not None:
            normalized_external_id = self.external_id.strip()
            self.external_id = normalized_external_id or None
        return self


class ManualStructurePreviewChapterResponse(BaseModel):
    """Lightweight projected chapter node for manual structure preview."""

    title: str
    external_id: str | None = None
    entry_index: int | None = Field(None, ge=0)
    sections: list[ManualStructurePreviewSectionResponse] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_preview_chapter(self) -> "ManualStructurePreviewChapterResponse":
        normalized_title = self.title.strip()
        if not normalized_title:
            raise ValueError("manual structure preview chapter title cannot be empty")
        self.title = normalized_title
        if self.external_id is not None:
            normalized_external_id = self.external_id.strip()
            self.external_id = normalized_external_id or None
        return self


class ManualStructurePreviewProvenanceResponse(BaseModel):
    """Lightweight parse provenance preview for manual structure validation."""

    parser_mode: str = Field(
        "manual_structure",
        description="Source-agnostic parser mode represented by the preview.",
    )
    source_hash: str | None = Field(
        None,
        description="Source hash used for stale-source preview checks when available.",
    )
    anchor_types: list[str] = Field(
        default_factory=list,
        description="Normalized anchor types observed in the previewed plan.",
    )

    @model_validator(mode="after")
    def _normalize_preview_provenance(self) -> "ManualStructurePreviewProvenanceResponse":
        normalized_parser_mode = self.parser_mode.strip().lower().replace("-", "_")
        if normalized_parser_mode != "manual_structure":
            raise ValueError("manual structure preview parser_mode must be manual_structure")
        self.parser_mode = normalized_parser_mode

        if self.source_hash is not None:
            normalized_source_hash = self.source_hash.strip()
            self.source_hash = normalized_source_hash or None

        normalized_anchor_types: list[str] = []
        for anchor_type in self.anchor_types:
            normalized_anchor_type = anchor_type.strip().lower().replace("-", "_")
            if normalized_anchor_type not in {"char_range", "page_range"}:
                raise ValueError("manual structure preview anchor_types must be char_range or page_range")
            normalized_anchor_types.append(normalized_anchor_type)
        self.anchor_types = normalized_anchor_types
        return self


class ManualStructureValidationResponse(BaseModel):
    """Response payload for manual structure validation/preview.

    This schema is response-only and intentionally lightweight: it previews the
    projected chapter/section shape without exposing raw text or task content.
    """

    doc_name: str
    valid: bool
    normalized_entries: list[ManualStructureNormalizedEntryResponse] = Field(
        default_factory=list
    )
    errors: list[ManualStructureValidationIssueResponse] = Field(default_factory=list)
    warnings: list[ManualStructureValidationIssueResponse] = Field(default_factory=list)
    preview_chapters: list[ManualStructurePreviewChapterResponse] = Field(default_factory=list)
    parse_provenance_preview: ManualStructurePreviewProvenanceResponse | None = None

    @model_validator(mode="after")
    def _normalize_validation_response(self) -> "ManualStructureValidationResponse":
        normalized_doc_name = self.doc_name.strip()
        if not normalized_doc_name:
            raise ValueError("doc_name cannot be empty")
        self.doc_name = normalized_doc_name
        if self.valid and self.errors:
            raise ValueError("manual structure validation response cannot be valid with errors")
        return self


class ReparseDocumentStructureRequest(BaseModel):
    """Request payload for explicit structure reparse action."""

    _ALLOWED_PARSER_MODES: ClassVar[frozenset[str]] = frozenset(
        {"common", "llm_enhanced", "manual_structure"}
    )

    doc_name: str = Field(..., description="Document name")
    parser_mode: str = Field(
        ...,
        description="Parser mode: common | llm_enhanced | manual_structure",
    )
    manual_structure: ManualStructurePlanRequest | None = Field(
        None,
        description=(
            "Required when parser_mode=manual_structure. "
            "Ignored by no other parser mode."
        ),
    )

    @model_validator(mode="after")
    def _normalize_reparse_request(self) -> "ReparseDocumentStructureRequest":
        normalized_doc_name = self.doc_name.strip()
        if not normalized_doc_name:
            raise ValueError("doc_name cannot be empty")
        self.doc_name = normalized_doc_name

        normalized_parser_mode = self.parser_mode.strip().lower().replace("-", "_")
        if normalized_parser_mode not in self._ALLOWED_PARSER_MODES:
            raise ValueError(
                "parser_mode must be common, llm_enhanced, or manual_structure"
            )
        self.parser_mode = normalized_parser_mode

        if normalized_parser_mode == "manual_structure":
            if self.manual_structure is None:
                raise ValueError("manual_structure is required when parser_mode is manual_structure")
            return self

        if self.manual_structure is not None:
            raise ValueError("manual_structure is only allowed when parser_mode is manual_structure")
        return self


class ReparseDocumentStructureResponse(BaseModel):
    """Response payload for structure reparse action."""

    success: bool
    doc_name: str
    parser_mode: str
    structured_document_path: str | None
    error: str | None
    section_count: int | None
