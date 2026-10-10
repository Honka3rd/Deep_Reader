from uuid import uuid4
from pathlib import Path
import re
import logging
import time

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import ValidationError

from app.qa_coordinator import QACoordinator
from app.section_task_coordinator import (
    ManualStructureAnchorDTO,
    ManualStructureEntryDTO,
    ManualStructurePlanDTO,
)
from logging_config import configure_logging
from document_structure.document_artifact_repository import DocumentListItem
from document_structure.manual_structure_projection import (
    ManualStructureAnchor,
    ManualStructureEntry,
    ManualStructureProjectionResult,
    project_manual_structure_plan,
)
from api_schemas import (
    ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS,
    AnalysisArtifactPayloadResponse,
    AnalysisInteractionGenerateRequest,
    AnalysisInteractionReadRequest,
    AnalysisInteractionRefreshRequest,
    AnalysisInteractionResponse,
    AnchorEvidenceResponse,
    ArtifactAwareInteractionMetadataResponse,
    ArtifactTargetRefResponse,
    ArtifactAvailabilityResponse,
    BatchTaskUnitContentRequest,
    BatchTaskUnitContentResponse,
    CriticalThinkingAnswerSubmitRequest,
    CriticalThinkingEvaluationRetryRequest,
    CriticalThinkingQuestionGenerateRequest,
    CriticalThinkingSessionPayloadResponse,
    CriticalThinkingSessionReadRequest,
    CriticalThinkingSessionResponse,
    DocumentTaskLayoutChapterResponse,
    ChapterQuizRequest,
    ChapterQuizResponse,
    PrepareDocumentRequest,
    PrepareDocumentResponse,
    PrepareTaskLayoutRequest,
    AskDocumentRequest,
    AskDocumentResponse,
    DocumentListItemResponse,
    DocumentListResponse,
    DocumentTaskLayoutResponse,
    EnhancedParseRecommendationResponse,
    GetDocumentTaskLayoutRequest,
    GetTaskUnitContentRequest,
    ManualStructureValidationIssueResponse,
    ManualStructureNormalizedEntryResponse,
    ManualStructurePreviewChapterResponse,
    ManualStructurePreviewProvenanceResponse,
    ManualStructurePreviewSectionResponse,
    ManualStructureValidationRequest,
    ManualStructureValidationResponse,
    ParseProvenanceResponse,
    ProfileStructureDiagnosticsResponse,
    QuizArtifactItemResponse,
    QuizArtifactPayloadResponse,
    QuizInteractionGenerateRequest,
    QuizInteractionReadRequest,
    QuizInteractionRefreshRequest,
    QuizInteractionResponse,
    QuizQuestionResponse,
    ReparseDocumentStructureRequest,
    ReparseDocumentStructureResponse,
    ReadingInteractionResponseEnvelope,
    ReadingInteractionTargetRequest,
    ReadingInteractionTargetResponse,
    SectionTaskLayoutResponse,
    SectionQuizResponse,
    SectionTaskRequest,
    SectionTaskResponse,
    SummarizeChapterRequest,
    SummarizeChapterResponse,
    TaskUnitContentBlockResponse,
    TaskUnitContentResponse,
    TaskUnitMetadataResponse,
    StatusResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

# 建立 FastAPI app
app = FastAPI(
    title="Deep Reader API",
    version="0.1.0",
)

# ⭐ QA coordinator 是自由問答主線 singleton
qa_coordinator = QACoordinator()
section_task_coordinator = qa_coordinator.container.section_task_coordinator()
document_artifact_repository = qa_coordinator.container.structured_document_artifact_repository()


def _should_suppress_success_request_log(path: str) -> bool:
    """Return whether successful request lifecycle logging should be suppressed."""
    return path == "/health"


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    """Log request lifecycle without recording request bodies or credentials."""
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - started_at) * 1000
    if not _should_suppress_success_request_log(request.url.path):
        logger.info(
            "request_completed method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
    return response

_RAW_DOCUMENT_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
_RAW_DOCUMENT_BASE_DIR = Path("data/raw")


def _raw_document_doc_name(path: Path, sibling_counts: dict[str, int]) -> str:
    """Return stable API doc_name for a raw file candidate."""
    if sibling_counts.get(path.stem, 0) > 1:
        return path.name
    return path.stem


def _list_raw_documents(
    query: str | None = None,
    limit: int = 200,
    base_dir: Path = _RAW_DOCUMENT_BASE_DIR,
) -> list[DocumentListItem]:
    """List raw document files as lightweight discovery candidates."""
    bounded_limit = max(1, min(limit, 200))
    normalized_query = (query or "").strip().casefold()
    if not base_dir.exists():
        return []

    raw_paths = [
        path
        for path in sorted(base_dir.iterdir(), key=lambda candidate: candidate.name.casefold())
        if path.is_file() and path.suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS
    ]
    sibling_counts: dict[str, int] = {}
    for path in raw_paths:
        sibling_counts[path.stem] = sibling_counts.get(path.stem, 0) + 1

    candidates: list[DocumentListItem] = []
    for path in raw_paths:
        doc_name = _raw_document_doc_name(path, sibling_counts)
        title = path.stem
        searchable = f"{doc_name} {title}".casefold()
        if normalized_query and normalized_query not in searchable:
            continue
        candidates.append(
            DocumentListItem(
                doc_name=doc_name,
                title=title,
                source=f"raw{path.suffix.casefold()}",
            )
        )
        if len(candidates) >= bounded_limit:
            break
    return candidates


def _merge_document_candidates(
    structured_items: list[DocumentListItem],
    raw_items: list[DocumentListItem],
    limit: int,
) -> list[DocumentListItem]:
    """Merge structured and raw candidates without duplicating the same doc_name."""
    bounded_limit = max(1, min(limit, 200))
    merged_by_name: dict[str, DocumentListItem] = {}

    for item in structured_items:
        merged_by_name[item.doc_name] = item

    for item in raw_items:
        existing = merged_by_name.get(item.doc_name)
        if existing is None:
            merged_by_name[item.doc_name] = item
            continue
        existing_sources = {
            source.strip()
            for source in existing.source.split("+")
            if source.strip()
        }
        existing_sources.add(item.source)
        merged_by_name[item.doc_name] = DocumentListItem(
            doc_name=existing.doc_name,
            title=existing.title or item.title,
            source="+".join(sorted(existing_sources)),
        )

    return sorted(
        merged_by_name.values(),
        key=lambda item: item.doc_name.casefold(),
    )[:bounded_limit]


def _filter_artifact_target_metadata(metadata: object) -> dict[str, object] | None:
    """Filter artifact-target metadata to approved glossary keys only."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("artifact target metadata must be a dictionary when provided")
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key) in ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS
    }


def _map_artifact_target_ref_response(target_ref: object) -> ArtifactTargetRefResponse:
    """Map shared artifact target reference into stable API response schema."""
    target_level_raw = getattr(target_ref, "target_level", None)
    target_level = (
        target_level_raw.value
        if hasattr(target_level_raw, "value")
        else str(target_level_raw)
    )
    return ArtifactTargetRefResponse(
        target_level=target_level,
        document_id=getattr(target_ref, "document_id", None),
        chapter_id=getattr(target_ref, "chapter_id", None),
        section_id=getattr(target_ref, "section_id", None),
        task_unit_id=getattr(target_ref, "task_unit_id", None),
        content_block_id=getattr(target_ref, "content_block_id", None),
        metadata=_filter_artifact_target_metadata(getattr(target_ref, "metadata", None)),
    )


def _map_task_unit_content_response(
    payload: object,
    *,
    include_raw_content: bool,
) -> TaskUnitContentResponse:
    """Map coordinator task-unit content DTO into public API response schema."""
    return TaskUnitContentResponse(
        document_id=payload.document_id,
        document_title=payload.document_title,
        task_unit_id=payload.task_unit_id,
        title=payload.title,
        container_title=payload.container_title,
        content=payload.content if include_raw_content else None,
        content_blocks=[
            TaskUnitContentBlockResponse(
                block_id=content_block.block_id,
                content=content_block.content,
                block_type=content_block.block_type,
                artifact_ids=(
                    None
                    if content_block.artifact_ids is None
                    else list(content_block.artifact_ids)
                ),
                artifact_target_refs=(
                    None
                    if content_block.artifact_target_refs is None
                    else [
                        _map_artifact_target_ref_response(target_ref)
                        for target_ref in content_block.artifact_target_refs
                    ]
                ),
                metadata=(
                    None
                    if content_block.metadata is None
                    else dict(content_block.metadata)
                ),
            )
            for content_block in payload.content_blocks
        ],
        source_section_ids=list(payload.source_section_ids),
        parent_section_id=payload.parent_section_id,
        section_id=payload.section_id,
        section_title=payload.section_title,
        chapter_id=payload.chapter_id,
        chapter_title=payload.chapter_title,
        is_fallback_generated=payload.is_fallback_generated,
    )


def _map_manual_structure_entry(entry) -> ManualStructureEntry:
    """Map API manual-structure entry into document_structure projector DTO."""
    return ManualStructureEntry(
        title=entry.title,
        level=entry.level,
        external_id=entry.external_id,
        notes=entry.notes,
        anchor=ManualStructureAnchor(
            anchor_type=entry.anchor.anchor_type,
            char_start=entry.anchor.char_start,
            char_end=entry.anchor.char_end,
            page_start_index=entry.anchor.page_start_index,
            page_end_index=entry.anchor.page_end_index,
        ),
    )


def _map_manual_structure_anchor_response(anchor: ManualStructureAnchor):
    """Map projector anchor into existing API anchor schema."""
    return {
        "anchor_type": anchor.anchor_type,
        "char_start": anchor.char_start,
        "char_end": anchor.char_end,
        "page_start_index": anchor.page_start_index,
        "page_end_index": anchor.page_end_index,
    }


def _manual_structure_issue_response(issue) -> ManualStructureValidationIssueResponse:
    """Map projector issue into public validation issue response."""
    return ManualStructureValidationIssueResponse(
        code=issue.code,
        severity=issue.severity,
        message=issue.message,
        entry_index=issue.entry_index,
    )


def _map_manual_structure_projection_response(
    request: ManualStructureValidationRequest,
    projection: ManualStructureProjectionResult,
) -> ManualStructureValidationResponse:
    """Map document_structure projector result into public validation response."""
    entries_by_index = {
        entry.entry_index: entry for entry in projection.normalized_entries
    }
    anchor_types: list[str] = []
    for entry in projection.normalized_entries:
        if entry.anchor.anchor_type not in anchor_types:
            anchor_types.append(entry.anchor.anchor_type)

    errors = [
        _manual_structure_issue_response(issue)
        for issue in projection.issues
        if issue.severity == "error"
    ]
    warnings = [
        _manual_structure_issue_response(issue)
        for issue in projection.issues
        if issue.severity != "error"
    ]
    return ManualStructureValidationResponse(
        doc_name=request.doc_name,
        valid=projection.valid,
        normalized_entries=[
            ManualStructureNormalizedEntryResponse(
                title=entry.title,
                level=entry.level,
                anchor=_map_manual_structure_anchor_response(entry.anchor),
                external_id=entry.external_id,
                projected_char_start=entry.anchor.char_start,
                projected_char_end=entry.anchor.char_end,
                projected_page_start_index=entry.anchor.page_start_index,
                projected_page_end_index=entry.anchor.page_end_index,
            )
            for entry in projection.normalized_entries
        ],
        errors=errors,
        warnings=warnings,
        preview_chapters=[
            ManualStructurePreviewChapterResponse(
                title=chapter.title,
                external_id=(
                    entries_by_index.get(chapter.source_entry_index).external_id
                    if entries_by_index.get(chapter.source_entry_index) is not None
                    else None
                ),
                entry_index=chapter.source_entry_index,
                sections=[
                    ManualStructurePreviewSectionResponse(
                        title=section.title,
                        external_id=(
                            entries_by_index.get(section.source_entry_index).external_id
                            if entries_by_index.get(section.source_entry_index) is not None
                            else None
                        ),
                        entry_index=section.source_entry_index,
                    )
                    for section in chapter.sections
                ],
            )
            for chapter in projection.preview_chapters
        ],
        parse_provenance_preview=ManualStructurePreviewProvenanceResponse(
            parser_mode="manual_structure",
            source_hash=projection.source_hash,
            anchor_types=anchor_types,
        ),
    )


def _build_manual_structure_validation_response(
    request: ManualStructureValidationRequest,
) -> ManualStructureValidationResponse:
    """Build a non-mutating preview through the document_structure projector."""
    projection = project_manual_structure_plan(
        [
            _map_manual_structure_entry(entry)
            for entry in request.manual_structure.entries
        ],
        source_hash=request.manual_structure.source_hash,
    )
    return _map_manual_structure_projection_response(request, projection)


def _map_manual_structure_plan_dto(
    request: ManualStructureValidationRequest | ReparseDocumentStructureRequest,
) -> ManualStructurePlanDTO:
    """Map public manual structure schema into app-layer coordinator DTO."""
    manual_structure = request.manual_structure
    if manual_structure is None:
        raise ValueError("manual_structure is required")
    return ManualStructurePlanDTO(
        source_hash=manual_structure.source_hash,
        entries=[
            ManualStructureEntryDTO(
                title=entry.title,
                level=entry.level,
                external_id=entry.external_id,
                anchor=ManualStructureAnchorDTO(
                    anchor_type=entry.anchor.anchor_type,
                    char_start=entry.anchor.char_start,
                    char_end=entry.anchor.char_end,
                    page_start_index=entry.anchor.page_start_index,
                    page_end_index=entry.anchor.page_end_index,
                ),
            )
            for entry in manual_structure.entries
        ],
    )


def _resolve_section_task_failure_status(reason: str) -> int:
    """Convert section-task failure reason into HTTP status code."""
    normalized_reason = reason.strip() or "section task failed"
    lowered = normalized_reason.lower()

    status_match = re.search(r"status=(\d{3})", normalized_reason)
    if status_match is not None:
        status_code = int(status_match.group(1))
        if 400 <= status_code <= 599:
            return status_code

    if "not found" in lowered:
        return 404
    return 400


def _resolve_reparse_failure_status(error: str) -> int:
    """Convert reparse failure reason into HTTP status code."""
    normalized_error = error.strip() or "reparse failed"
    lowered = normalized_error.lower()

    status_match = re.search(r"status=(\d{3})", normalized_error)
    if status_match is not None:
        status_code = int(status_match.group(1))
        if 400 <= status_code <= 599:
            return status_code

    if (
        "bad_request" in lowered
        or "cannot be empty" in lowered
        or "unsupported" in lowered
        or "unknown parser_mode" in lowered
    ):
        return 400
    return 500


def _resolve_task_unit_content_failure_status(reason: str) -> int:
    """Convert task-unit content lookup failures into HTTP status code."""
    normalized_reason = reason.strip() or "task unit content lookup failed"
    lowered = normalized_reason.lower()
    if "not found" in lowered:
        return 404
    return 400


_READING_INTERACTION_STATUS_HTTP_STATUS: dict[str, int] = {
    "not_generated": 200,
    "completed": 200,
    "insufficient_content": 200,
    "question_generated": 200,
    "answer_submitted": 200,
    "evaluation_failed": 200,
    "stale_target": 409,
    "validation_failed": 422,
    "generation_failed": 502,
}


def _reading_interaction_http_status(status: str) -> int:
    """Map interaction envelope status to an HTTP status for public routes."""
    return _READING_INTERACTION_STATUS_HTTP_STATUS.get(status.strip().lower(), 500)


def _reading_interaction_exception_status(error: Exception) -> int:
    """Map route-layer interaction exceptions to stable HTTP statuses."""
    if isinstance(error, FileNotFoundError):
        return 404
    if isinstance(error, ValidationError):
        return 422
    if isinstance(error, ValueError):
        message = str(error).lower()
        if "not found" in message:
            return 404
        if "stale" in message or "conflict" in message:
            return 409
        return 400
    return 500


def _raise_reading_interaction_http_exception(error: Exception) -> None:
    """Raise a public HTTP error for reading interaction routes."""
    raise HTTPException(
        status_code=_reading_interaction_exception_status(error),
        detail=str(error),
    )


def _reading_target_kwargs(
    target: ReadingInteractionTargetRequest,
) -> dict[str, str | None]:
    """Map public reading target schema into app-layer keyword args."""
    return {
        "doc_name": target.doc_name,
        "target_level": target.target_type,
        "chapter_id": target.chapter_id,
        "section_id": target.section_id,
        "task_unit_id": target.task_unit_id,
    }


def _map_artifact_context_metadata(
    metadata: dict[str, object],
) -> ArtifactAwareInteractionMetadataResponse | None:
    """Map persisted artifact-reference metadata into public response metadata."""
    raw_reference = metadata.get("artifact_reference")
    if not isinstance(raw_reference, dict):
        return None
    return ArtifactAwareInteractionMetadataResponse(
        artifact_context_mode=str(raw_reference.get("artifact_context_mode", "none")),
        primary_source_evidence_ids=[
            str(value)
            for value in raw_reference.get("primary_source_evidence_ids", [])
        ],
        referenced_artifact_ids=[
            str(value)
            for value in raw_reference.get("referenced_artifact_ids", [])
        ],
        referenced_artifact_types=[
            str(value)
            for value in raw_reference.get("referenced_artifact_types", [])
        ],
        referenced_artifact_target_levels=[
            str(value)
            for value in raw_reference.get("referenced_artifact_target_levels", [])
        ],
        coverage_counts=dict(raw_reference.get("coverage_counts", {})),
        deduplication_hint_applied=bool(
            raw_reference.get("deduplication_hint_applied", False)
        ),
        abstraction_hint_applied=bool(
            raw_reference.get("abstraction_hint_applied", False)
        ),
        artifact_context_pruned_reason=(
            None
            if raw_reference.get("artifact_context_pruned_reason") is None
            else str(raw_reference.get("artifact_context_pruned_reason"))
        ),
    )


def _map_reading_interaction_envelope(result) -> ReadingInteractionResponseEnvelope:
    """Map app reading-interaction DTO into the shared public envelope."""
    target = result.target
    metadata = dict(result.metadata)
    return ReadingInteractionResponseEnvelope(
        target=ReadingInteractionTargetResponse(
            doc_name=target.doc_name,
            target_type=target.target_level,
            target_id=target.target_id,
            document_id=target.document_id,
            document_title=target.document_title,
            chapter_id=target.chapter_id,
            section_id=target.section_id,
            task_unit_id=target.task_unit_id,
            title=target.title,
        ),
        interaction_type=result.interaction_type,
        status=result.status,
        artifact_id=result.artifact_id,
        session_id=result.session_id,
        schema_version=(
            None
            if metadata.get("output_schema_version") is None
            else str(metadata.get("output_schema_version"))
        ),
        prompt_instruction_version=(
            None
            if metadata.get("prompt_instruction_version") is None
            else str(metadata.get("prompt_instruction_version"))
        ),
        reason=result.reason,
        artifact_context_metadata=_map_artifact_context_metadata(metadata),
    )


def _map_analysis_interaction_response(result) -> AnalysisInteractionResponse:
    """Map app analysis result into the public analysis response schema."""
    payload = None
    if result.status == "completed":
        raw_payload = dict(result.payload)
        interpretation = raw_payload.get("interpretation")
        if not isinstance(interpretation, str) or not interpretation.strip():
            interpretation = raw_payload.get("reasoning")
        payload = AnalysisArtifactPayloadResponse(
            summary=str(raw_payload.get("summary", "")),
            reasoning=str(raw_payload.get("reasoning", "")),
            interpretation=str(interpretation or ""),
            explanation=(
                None
                if raw_payload.get("explanation") is None
                else str(raw_payload.get("explanation"))
            ),
            key_points=[
                str(key_point)
                for key_point in raw_payload.get("key_points", [])
            ],
        )
    return AnalysisInteractionResponse(
        envelope=_map_reading_interaction_envelope(result),
        payload=payload,
    )


def _map_quiz_interaction_response(result) -> QuizInteractionResponse:
    """Map app quiz result into the public quiz response schema."""
    payload = None
    if result.status == "completed":
        raw_payload = dict(result.payload)
        raw_items = raw_payload.get("items", [])
        payload = QuizArtifactPayloadResponse(
            max_items=(
                None
                if result.metadata.get("max_items") is None
                else int(result.metadata.get("max_items"))
            ),
            items=[
                _map_quiz_interaction_item(raw_item, index)
                for index, raw_item in enumerate(raw_items)
            ],
        )
    return QuizInteractionResponse(
        envelope=_map_reading_interaction_envelope(result),
        payload=payload,
    )


def _map_quiz_interaction_item(
    raw_item: object,
    index: int,
) -> QuizArtifactItemResponse:
    """Map internal quiz item keys to the drawer-oriented public schema."""
    if not isinstance(raw_item, dict):
        raise ValueError(f"quiz item {index} must be an object")
    item_id = raw_item.get("item_id")
    item_type = raw_item.get("item_type", raw_item.get("type"))
    prompt = raw_item.get("prompt", raw_item.get("question"))
    options = raw_item.get("options", raw_item.get("choices"))
    return QuizArtifactItemResponse(
        item_id=str(item_id or f"q{index + 1}"),
        item_type=str(item_type or ""),
        prompt=str(prompt or ""),
        options=None if options is None else [str(option) for option in options],
        answer=raw_item.get("answer"),
        explanation=(
            None
            if raw_item.get("explanation") is None
            else str(raw_item.get("explanation"))
        ),
    )


def _map_critical_thinking_session_response(result) -> CriticalThinkingSessionResponse:
    """Map app critical-thinking session result into the public response schema."""
    payload = None
    if result.status in {
        "question_generated",
        "answer_submitted",
        "evaluation_failed",
        "completed",
    }:
        raw_payload = dict(result.payload)
        submitted_answer = raw_payload.get(
            "submitted_answer",
            raw_payload.get("answer"),
        )
        payload = CriticalThinkingSessionPayloadResponse(
            question=(
                None
                if raw_payload.get("question") is None
                else str(raw_payload.get("question"))
            ),
            submitted_answer=(
                None
                if submitted_answer is None
                else str(submitted_answer)
            ),
            evaluation=raw_payload.get("evaluation"),
        )
    return CriticalThinkingSessionResponse(
        envelope=_map_reading_interaction_envelope(result),
        payload=payload,
    )

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health", response_model=StatusResponse)
def health():
    """Return API health status.

Returns:
    Service health payload used by readiness checks."""
    return StatusResponse(
        status="ok",
        message="Deep Reader API is running",
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(
    q: str | None = Query(
        None,
        description="Optional case-insensitive substring query against document name/title.",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Maximum number of document candidates to return.",
    ),
):
    """List lightweight document candidates for UI document search."""
    normalized_query = None if q is None or not q.strip() else q.strip()
    try:
        items = document_artifact_repository.list_documents(
            query=normalized_query,
            limit=200,
        )
        raw_items = _list_raw_documents(
            query=normalized_query,
            limit=200,
        )
        merged_items = _merge_document_candidates(
            structured_items=items,
            raw_items=raw_items,
            limit=limit,
        )
        response_items = [
            DocumentListItemResponse(
                doc_name=item.doc_name,
                title=item.title,
                source=item.source,
            )
            for item in merged_items
        ]
        return DocumentListResponse(
            items=response_items,
            query=normalized_query,
            total=len(response_items),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/prepare", response_model=PrepareDocumentResponse)
def prepare_document(request: PrepareDocumentRequest, response: Response):
    """Prepare document artifacts through REST for deterministic parser-mode testing."""
    try:
        assets = qa_coordinator.document_preparation_pipeline.prepare(
            doc_name=request.doc_name,
            force_rebuild=request.force_rebuild,
            mode=request.mode,
            structured_parser_mode=request.structured_parser_mode,
        )

        success = (
            assets.structured_document_ready
            and (
                request.mode.strip().lower() == "base"
                or (assets.faiss_ready and assets.profile_ready and assets.bundle_ready)
            )
        )
        if not success:
            response.status_code = 400

        return PrepareDocumentResponse(
            doc_name=request.doc_name,
            mode=request.mode,
            structured_parser_mode=request.structured_parser_mode,
            success=success,
            structured_document_ready=assets.structured_document_ready,
            structured_document_path=assets.structured_document_path,
            faiss_ready=assets.faiss_ready,
            profile_ready=assets.profile_ready,
            bundle_ready=assets.bundle_ready,
            errors=list(assets.errors),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/prepare-task-layout", response_model=DocumentTaskLayoutResponse)
def prepare_task_layout(request: PrepareTaskLayoutRequest):
    """Prepare a document when needed, then return its task-layout projection."""
    try:
        assets = qa_coordinator.document_preparation_pipeline.prepare(
            doc_name=request.doc_name,
            force_rebuild=request.force_rebuild,
            mode="base",
            structured_parser_mode=request.structured_parser_mode,
        )
        if not assets.structured_document_ready:
            detail = " | ".join(assets.errors) or "structured document preparation failed"
            raise HTTPException(status_code=400, detail=detail)

        return get_document_task_layout(
            GetDocumentTaskLayoutRequest(
                doc_name=request.doc_name,
                refresh_task_units=request.refresh_task_units,
                task_unit_split_mode=request.task_unit_split_mode,
                semantic_top_k_candidates=request.semantic_top_k_candidates,
                include_anchor_page_evidence=request.include_anchor_page_evidence,
            )
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# ---------------------------
# Ask Question
# ---------------------------
@app.post("/documents/ask", response_model=AskDocumentResponse)
def ask_document(request: AskDocumentRequest, response: Response):
    """Execute `/documents/ask` request and return answer payload.

Args:
    request: API request payload model.

Returns:
    QA response payload including answer text and effective session id."""
    try:
        session_id = request.session_id.strip() if request.session_id else ""
        if not session_id:
            session_id = str(uuid4())

        ask_result = qa_coordinator.ask(
            doc_name=request.doc_name,
            question=request.query,
            top_k=request.top_k,
            session_id=session_id,
        )
        response.status_code = 201 if ask_result.is_low_value else 200

        return AskDocumentResponse(
            doc_name=request.doc_name,
            query=request.query,
            answer=ask_result.answer_text,
            session_id=session_id,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/documents/reading-interactions/insight/read",
    response_model=AnalysisInteractionResponse,
)
def read_analysis_interaction(
    request: AnalysisInteractionReadRequest,
    response: Response,
):
    """Read a persisted inline insight/analysis artifact without generation."""
    try:
        result = section_task_coordinator.read_analysis_artifact(
            **_reading_target_kwargs(request.target)
        )
        mapped_response = _map_analysis_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/insight/generate",
    response_model=AnalysisInteractionResponse,
)
def generate_analysis_interaction(
    request: AnalysisInteractionGenerateRequest,
    response: Response,
):
    """Generate or reuse a persisted inline insight/analysis artifact."""
    try:
        result = section_task_coordinator.generate_analysis_artifact(
            **_reading_target_kwargs(request.target),
            prompt_instruction_version=request.prompt_instruction_version,
        )
        mapped_response = _map_analysis_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/insight/refresh",
    response_model=AnalysisInteractionResponse,
)
def refresh_analysis_interaction(
    request: AnalysisInteractionRefreshRequest,
    response: Response,
):
    """Explicitly regenerate the current inline insight/analysis artifact."""
    try:
        result = section_task_coordinator.refresh_analysis_artifact(
            **_reading_target_kwargs(request.target),
            prompt_instruction_version=request.prompt_instruction_version,
        )
        mapped_response = _map_analysis_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/quiz/read",
    response_model=QuizInteractionResponse,
)
def read_quiz_interaction(
    request: QuizInteractionReadRequest,
    response: Response,
):
    """Read a persisted target-agnostic quiz artifact without generation."""
    try:
        result = section_task_coordinator.read_quiz_artifact(
            **_reading_target_kwargs(request.target)
        )
        mapped_response = _map_quiz_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/quiz/generate",
    response_model=QuizInteractionResponse,
)
def generate_quiz_interaction(
    request: QuizInteractionGenerateRequest,
    response: Response,
):
    """Generate or reuse a persisted target-agnostic quiz artifact."""
    try:
        result = section_task_coordinator.generate_quiz_artifact(
            **_reading_target_kwargs(request.target),
            prompt_instruction_version=request.prompt_instruction_version,
        )
        mapped_response = _map_quiz_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/quiz/refresh",
    response_model=QuizInteractionResponse,
)
def refresh_quiz_interaction(
    request: QuizInteractionRefreshRequest,
    response: Response,
):
    """Explicitly regenerate the current target-agnostic quiz artifact."""
    try:
        result = section_task_coordinator.refresh_quiz_artifact(
            **_reading_target_kwargs(request.target),
            prompt_instruction_version=request.prompt_instruction_version,
        )
        mapped_response = _map_quiz_interaction_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/critical-thinking/read",
    response_model=CriticalThinkingSessionResponse,
)
def read_critical_thinking_interaction(
    request: CriticalThinkingSessionReadRequest,
    response: Response,
):
    """Read a persisted critical-thinking session without generation."""
    try:
        result = section_task_coordinator.read_critical_thinking_session(
            **_reading_target_kwargs(request.target),
            session_id=request.session_id,
        )
        mapped_response = _map_critical_thinking_session_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/critical-thinking/generate-question",
    response_model=CriticalThinkingSessionResponse,
)
def generate_critical_thinking_question_interaction(
    request: CriticalThinkingQuestionGenerateRequest,
    response: Response,
):
    """Generate and persist one critical-thinking question session."""
    try:
        result = section_task_coordinator.generate_critical_thinking_question(
            **_reading_target_kwargs(request.target),
            prompt_instruction_version=request.prompt_instruction_version,
        )
        mapped_response = _map_critical_thinking_session_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/critical-thinking/submit-answer",
    response_model=CriticalThinkingSessionResponse,
)
def submit_critical_thinking_answer_interaction(
    request: CriticalThinkingAnswerSubmitRequest,
    response: Response,
):
    """Submit an answer and evaluate the critical-thinking session."""
    try:
        result = section_task_coordinator.submit_critical_thinking_answer(
            **_reading_target_kwargs(request.target),
            session_id=request.session_id,
            answer=request.answer,
        )
        mapped_response = _map_critical_thinking_session_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post(
    "/documents/reading-interactions/critical-thinking/retry-evaluation",
    response_model=CriticalThinkingSessionResponse,
)
def retry_critical_thinking_evaluation_interaction(
    request: CriticalThinkingEvaluationRetryRequest,
    response: Response,
):
    """Retry evaluation while preserving the generated question and answer."""
    try:
        result = section_task_coordinator.retry_critical_thinking_evaluation(
            **_reading_target_kwargs(request.target),
            session_id=request.session_id,
        )
        mapped_response = _map_critical_thinking_session_response(result)
        response.status_code = _reading_interaction_http_status(
            mapped_response.envelope.status
        )
        return mapped_response
    except (FileNotFoundError, ValueError, ValidationError) as error:
        _raise_reading_interaction_http_exception(error)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post("/documents/section-summary", response_model=SectionTaskResponse)
def summarize_document_section(request: SectionTaskRequest, response: Response):
    """Run summary task for one structured section."""
    try:
        result = section_task_coordinator.summarize_section(
            doc_name=request.doc_name,
            section_id=request.section_id,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
            refresh_summary=request.refresh_summary,
        )
        if result.success:
            return SectionTaskResponse(
                doc_name=request.doc_name,
                section_id=request.section_id,
                success=True,
                result=result.payload,
                reason=None,
                cache_hit=result.cache_hit,
            )
        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SectionTaskResponse(
            doc_name=request.doc_name,
            section_id=request.section_id,
            success=False,
            result=None,
            reason=result.reason,
            cache_hit=result.cache_hit,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/section-quiz", response_model=SectionQuizResponse)
def generate_document_section_quiz(request: SectionTaskRequest, response: Response):
    """Run quiz task for one structured section."""
    try:
        result = section_task_coordinator.generate_section_quiz(
            doc_name=request.doc_name,
            section_id=request.section_id,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
            refresh_quiz=request.refresh_quiz,
        )
        if result.success:
            questions = [
                QuizQuestionResponse(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    answer_text=question.answer_text,
                )
                for question in (result.payload or [])
            ]
            return SectionQuizResponse(
                doc_name=request.doc_name,
                section_id=request.section_id,
                success=True,
                questions=questions,
                reason=None,
                cache_hit=result.cache_hit,
            )
        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SectionQuizResponse(
            doc_name=request.doc_name,
            section_id=request.section_id,
            success=False,
            questions=None,
            reason=result.reason,
            cache_hit=result.cache_hit,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/summarize-chapter", response_model=SummarizeChapterResponse)
def summarize_document_chapter(
    request: SummarizeChapterRequest,
    response: Response,
):
    """Run summary task for one chapter resolved by chapter_id first, then title."""
    try:
        response_chapter_title = request.chapter_title or request.chapter_id or ""
        result = section_task_coordinator.summarize_chapter(
            doc_name=request.doc_name,
            chapter_title=request.chapter_title,
            chapter_id=request.chapter_id,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
            refresh_summary=request.refresh_summary,
        )
        if result.success:
            return SummarizeChapterResponse(
                doc_name=request.doc_name,
                chapter_title=response_chapter_title,
                success=True,
                result=result.payload,
                reason=None,
                cache_hit=result.cache_hit,
            )

        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SummarizeChapterResponse(
            doc_name=request.doc_name,
            chapter_title=response_chapter_title,
            success=False,
            result=None,
            reason=result.reason,
            cache_hit=result.cache_hit,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/chapter-quiz", response_model=ChapterQuizResponse)
def generate_document_chapter_quiz(
    request: ChapterQuizRequest,
    response: Response,
):
    """Run quiz task for one chapter resolved by chapter_id first, then title."""
    try:
        response_chapter_title = request.chapter_title or request.chapter_id or ""
        result = section_task_coordinator.generate_chapter_quiz(
            doc_name=request.doc_name,
            chapter_title=request.chapter_title,
            chapter_id=request.chapter_id,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
            refresh_quiz=request.refresh_quiz,
        )
        if result.success:
            questions = [
                QuizQuestionResponse(
                    question_id=question.question_id,
                    question_text=question.question_text,
                    answer_text=question.answer_text,
                )
                for question in (result.payload or [])
            ]
            return ChapterQuizResponse(
                doc_name=request.doc_name,
                chapter_title=response_chapter_title,
                success=True,
                questions=questions,
                reason=None,
                cache_hit=result.cache_hit,
            )

        response.status_code = _resolve_section_task_failure_status(result.reason)
        return ChapterQuizResponse(
            doc_name=request.doc_name,
            chapter_title=response_chapter_title,
            success=False,
            questions=None,
            reason=result.reason,
            cache_hit=result.cache_hit,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/task-layout", response_model=DocumentTaskLayoutResponse)
def get_document_task_layout(request: GetDocumentTaskLayoutRequest):
    """Read current effective document-task layout snapshot."""
    try:
        layout = section_task_coordinator.get_document_task_layout(
            doc_name=request.doc_name,
            refresh_task_units=request.refresh_task_units,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
            include_anchor_page_evidence=request.include_anchor_page_evidence,
        )
        def _artifact_response(
            artifact,
        ) -> ArtifactAvailabilityResponse | None:
            if artifact is None:
                return None
            return ArtifactAvailabilityResponse(
                has_summary=artifact.has_summary,
                has_quiz=artifact.has_quiz,
                summary_cache_valid=artifact.summary_cache_valid,
                quiz_cache_valid=artifact.quiz_cache_valid,
                summary_invalid_reason=artifact.summary_invalid_reason,
                quiz_invalid_reason=artifact.quiz_invalid_reason,
                summary_generated_at=artifact.summary_generated_at,
                quiz_generated_at=artifact.quiz_generated_at,
            )

        def _task_unit_response(task_unit) -> TaskUnitMetadataResponse:
            return TaskUnitMetadataResponse(
                unit_id=task_unit.unit_id,
                title=task_unit.title,
                container_title=task_unit.container_title,
                source_section_ids=list(task_unit.source_section_ids),
                is_fallback_generated=task_unit.is_fallback_generated,
                artifacts=_artifact_response(task_unit.artifacts),
            )

        def _anchor_evidence_response(anchor_evidence) -> AnchorEvidenceResponse | None:
            if anchor_evidence is None:
                return None
            return AnchorEvidenceResponse(
                anchor_type=anchor_evidence.anchor_type,
                status=anchor_evidence.status,
                reason=anchor_evidence.reason,
                char_start=anchor_evidence.char_start,
                char_end=anchor_evidence.char_end,
                page_start_index=anchor_evidence.page_start_index,
                page_end_index=anchor_evidence.page_end_index,
                page_start_label=anchor_evidence.page_start_label,
                page_end_label=anchor_evidence.page_end_label,
            )

        def _section_response(section) -> SectionTaskLayoutResponse:
            return SectionTaskLayoutResponse(
                section_id=section.section_id,
                title=section.title,
                container_title=section.container_title,
                section_role=section.section_role,
                parent_chapter_id=section.parent_chapter_id,
                section_kind=section.section_kind,
                is_implicit_section=section.is_implicit_section,
                task_mode=section.task_mode.value,
                task_units=[
                    _task_unit_response(task_unit)
                    for task_unit in section.task_units
                ],
                artifacts=_artifact_response(section.artifacts),
                anchor_evidence=_anchor_evidence_response(section.anchor_evidence),
            )

        chapters = [
            DocumentTaskLayoutChapterResponse(
                chapter_id=chapter.chapter_id,
                title=chapter.title,
                level=chapter.level,
                chapter_role=chapter.chapter_role,
                sections=[_section_response(section) for section in chapter.sections],
                artifacts=_artifact_response(chapter.artifacts),
                anchor_evidence=_anchor_evidence_response(chapter.anchor_evidence),
                metadata=dict(chapter.metadata),
            )
            for chapter in layout.chapters
        ]
        recommendation = None
        if layout.enhanced_parse_recommendation is not None:
            recommendation = EnhancedParseRecommendationResponse(
                should_recommend=layout.enhanced_parse_recommendation.should_recommend,
                score=layout.enhanced_parse_recommendation.score,
                reasons=list(layout.enhanced_parse_recommendation.reasons),
                metrics=dict(layout.enhanced_parse_recommendation.metrics),
            )

        return DocumentTaskLayoutResponse(
            document_id=layout.document_id,
            title=layout.title,
            language=layout.language,
            chapters=chapters,
            enhanced_parse_recommendation=recommendation,
            profile_diagnostics=(
                None
                if layout.profile_diagnostics is None
                else ProfileStructureDiagnosticsResponse(
                    parser_metadata_shape=layout.profile_diagnostics.parser_metadata_shape,
                    post_actual_structure_shape=layout.profile_diagnostics.post_actual_structure_shape,
                    title_uniqueness_risk=layout.profile_diagnostics.title_uniqueness_risk,
                    title_target_requires_id=layout.profile_diagnostics.title_target_requires_id,
                    task_unit_stats_available=layout.profile_diagnostics.task_unit_stats_available,
                    task_unit_section_coverage=layout.profile_diagnostics.task_unit_section_coverage,
                    parser_post_shape_mismatch=layout.profile_diagnostics.parser_post_shape_mismatch,
                    enhanced_parse_hint=layout.profile_diagnostics.enhanced_parse_hint,
                    warnings=list(layout.profile_diagnostics.warnings),
                )
            ),
            parse_provenance=(
                None
                if layout.parse_provenance is None
                else ParseProvenanceResponse(
                    requested_parser_mode=layout.parse_provenance.requested_parser_mode,
                    effective_parser_mode=layout.parse_provenance.effective_parser_mode,
                    fallback_used=layout.parse_provenance.fallback_used,
                    fallback_reason=layout.parse_provenance.fallback_reason,
                    source=layout.parse_provenance.source,
                )
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get(
    "/documents/{doc_name}/task-units/{task_unit_id}/content",
    response_model=TaskUnitContentResponse,
)
def get_task_unit_content(
    doc_name: str,
    task_unit_id: str,
    segmented: bool = Query(
        False,
        description=(
            "When true, return explicit opt-in deterministic segmented content blocks. "
            "When false, preserve compatibility-safe default content-block behavior."
        ),
    ),
    include_raw_content: bool = Query(
        False,
        description=(
            "When true, include legacy raw task-unit content string for compatibility/debug. "
            "When false, return content-block-first payload without raw content duplication."
        ),
    ),
):
    """Read one task-unit render content payload on demand by task_unit_id."""
    try:
        request = GetTaskUnitContentRequest(
            doc_name=doc_name,
            task_unit_id=task_unit_id,
            segmented=segmented,
            include_raw_content=include_raw_content,
        )
        payload = section_task_coordinator.get_task_unit_content(
            doc_name=request.doc_name,
            task_unit_id=request.task_unit_id,
            segmented=request.segmented,
        )
        return _map_task_unit_content_response(
            payload,
            include_raw_content=request.include_raw_content,
        )
    except (ValueError, ValidationError) as error:
        raise HTTPException(
            status_code=_resolve_task_unit_content_failure_status(str(error)),
            detail=str(error),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post(
    "/documents/{doc_name}/task-units/content",
    response_model=BatchTaskUnitContentResponse,
)
def get_batch_task_unit_content(
    doc_name: str,
    request: BatchTaskUnitContentRequest,
):
    """Read ordered task-unit render content payloads with one request."""
    try:
        payloads = section_task_coordinator.get_task_unit_contents(
            doc_name=doc_name,
            task_unit_ids=request.task_unit_ids,
            segmented=request.segmented,
        )
        document_id = payloads[0].document_id if payloads else ""
        document_title = payloads[0].document_title if payloads else ""
        return BatchTaskUnitContentResponse(
            document_id=document_id,
            document_title=document_title,
            contents=[
                _map_task_unit_content_response(
                    payload,
                    include_raw_content=request.include_raw_content,
                )
                for payload in payloads
            ],
        )
    except (ValueError, ValidationError) as error:
        raise HTTPException(
            status_code=_resolve_task_unit_content_failure_status(str(error)),
            detail=str(error),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post(
    "/documents/manual-structure/validate",
    response_model=ManualStructureValidationResponse,
)
def validate_manual_structure(
    request: ManualStructureValidationRequest,
):
    """Validate and preview a manual structure plan without mutating document state."""
    try:
        return _build_manual_structure_validation_response(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/documents/reparse-structure", response_model=ReparseDocumentStructureResponse)
def reparse_document_structure(
    request: ReparseDocumentStructureRequest,
    response: Response,
):
    """Run explicit structure reparse action and replace single active source on success."""
    normalized_parser_mode = request.parser_mode.strip().lower().replace("-", "_")
    if normalized_parser_mode == "manual_structure":
        result = section_task_coordinator.commit_manual_structure_reparse(
            doc_name=request.doc_name,
            manual_structure=_map_manual_structure_plan_dto(request),
        )
        response.status_code = result.status_code
        return ReparseDocumentStructureResponse(
            success=result.success,
            doc_name=result.doc_name,
            parser_mode=result.parser_mode,
            structured_document_path=result.structured_document_path,
            error=result.error,
            section_count=result.section_count,
        )

    if normalized_parser_mode not in {"common", "llm_enhanced"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "unknown parser_mode. supported values: common, llm_enhanced"
            ),
        )

    try:
        result = section_task_coordinator.reparse_document_structure(
            doc_name=request.doc_name,
            parser_mode=normalized_parser_mode,
        )
        if result.success:
            return ReparseDocumentStructureResponse(
                success=True,
                doc_name=result.doc_name,
                parser_mode=result.parser_mode,
                structured_document_path=result.structured_document_path,
                error=None,
                section_count=result.section_count,
            )

        response.status_code = _resolve_reparse_failure_status(result.error or "")
        return ReparseDocumentStructureResponse(
            success=False,
            doc_name=result.doc_name,
            parser_mode=result.parser_mode,
            structured_document_path=result.structured_document_path,
            error=result.error,
            section_count=result.section_count,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
