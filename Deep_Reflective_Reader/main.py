from uuid import uuid4
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import logging
import shutil
import time

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from doc_loaders.pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason

from app.qa_coordinator import QACoordinator
from app.enhancement_preflight import EnhancementConflict, enhancement_preflight_service
from api_schemas import EnhancementPreflightRequest, EnhancementPreflightResponse
from api_schemas import EnhancementOperationRequest, EnhancementOperationResponse
from app.task_layout_readiness import TaskLayoutNotReadyError
from app.section_task_coordinator import (
    ManualStructureAnchorDTO,
    ManualStructureEntryDTO,
    ManualStructurePlanDTO,
)
from logging_config import configure_logging, log_stage, request_id_context
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
    PrepareTaskLayoutResponse,
    PdfExtractionFailureResponse,
    PdfExtractionMetadataResponse,
    PreparationValidationFailureResponse,
    AskDocumentRequest,
    AskDocumentResponse,
    DocumentListItemResponse,
    DocumentListResponse,
    DocumentLifecycleActionRequest,
    DocumentLifecycleActionResponse,
    DocumentManagementItemResponse,
    DocumentManagementListResponse,
    DocumentUploadResponse,
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
enhancement_preflight_service.reader = section_task_coordinator.read_enhancement_state
enhancement_preflight_service.lease = section_task_coordinator.enhancement_lease
enhancement_preflight_service.availability = section_task_coordinator.enhancement_unavailable_reason
enhancement_preflight_service.capacity_reader = section_task_coordinator.inspect_enhancement_capacity


def _reading_content_service():
    from document_structure.reading_content import ReadingContentService
    return ReadingContentService(document_artifact_repository,
        section_task_coordinator.document_preparation_pipeline.loader_factory)


def _reading_content_error(error):
    from document_structure.reading_content import ReadingContentConflict
    if isinstance(error,FileNotFoundError):
        return HTTPException(status_code=404,detail="document_or_region_missing")
    if isinstance(error,(ReadingContentConflict,ValueError)):
        return HTTPException(status_code=409,detail=str(error))
    logger.exception("reading_content_operation_failed")
    return HTTPException(status_code=503,detail="reading_content_unavailable")


from api_schemas import ReadingContentMutationRequest


@app.get("/documents/{doc_name}/reading-content")
def read_reading_content(doc_name: str, include_blocks: bool = False, block_id: str | None = None):
    try:
        result = _reading_content_service().read(doc_name)
        if block_id is not None:
            result["blocks"] = [b for b in result["blocks"] if b["block_id"]==block_id]
        elif not include_blocks:
            result["blocks"] = []
        return result
    except Exception as error:
        raise _reading_content_error(error) from error


@app.post("/documents/{doc_name}/reading-content")
def mutate_reading_content(doc_name: str, request: ReadingContentMutationRequest):
    try:
        result = _reading_content_service().mutate(doc_name,**request.model_dump())
        result["blocks"] = [b for b in result["blocks"] if b["block_id"]==request.block_id]
        return result
    except Exception as error:
        raise _reading_content_error(error) from error


@app.get("/documents/{doc_name}/source-review/{region_id}/image")
def preview_reading_source(doc_name: str, region_id: str):
    from fastapi.responses import Response
    try:
        return Response(_reading_content_service().preview(doc_name,region_id),media_type="image/png",
                        headers={"Cache-Control":"no-store"})
    except Exception as error:
        raise _reading_content_error(error) from error


def _should_suppress_success_request_log(path: str) -> bool:
    """Return whether successful request lifecycle logging should be suppressed."""
    return path == "/health"


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    """Log request lifecycle without recording request bodies or credentials."""
    started_at = time.perf_counter()
    request_id = uuid4().hex
    token = request_id_context.set(request_id)
    if not _should_suppress_success_request_log(request.url.path):
        logger.info("request_started method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if not _should_suppress_success_request_log(request.url.path):
            logger.info(
                "request_completed method=%s path=%s status=%s duration_ms=%.1f",
                request.method, request.url.path, response.status_code,
                (time.perf_counter() - started_at) * 1000,
            )
        return response
    except Exception:
        logger.exception("request_failed method=%s path=%s duration_ms=%.1f",
                         request.method, request.url.path,
                         (time.perf_counter() - started_at) * 1000)
        raise
    finally:
        request_id_context.reset(token)

_RAW_DOCUMENT_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
_RAW_DOCUMENT_BASE_DIR = Path("data/raw")
_DERIVED_DOCUMENT_BASE_DIR = Path("data/faiss")
_DOCUMENT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024


def _raw_document_doc_name(path: Path, sibling_counts: dict[str, int]) -> str:
    """Return stable API doc_name for a raw file candidate."""
    if sibling_counts.get(path.stem, 0) > 1:
        return path.name
    return path.stem


def _raw_document_sibling_counts(base_dir: Path = _RAW_DOCUMENT_BASE_DIR) -> dict[str, int]:
    """Return supported raw-file stem counts for doc_name disambiguation."""
    if not base_dir.exists():
        return {}
    sibling_counts: dict[str, int] = {}
    for path in base_dir.iterdir():
        if path.is_file() and path.suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS:
            sibling_counts[path.stem] = sibling_counts.get(path.stem, 0) + 1
    return sibling_counts


def _normalize_upload_filename(filename: str | None) -> str:
    """Validate and normalize a user-uploaded raw source filename."""
    normalized = (filename or "").strip()
    if (
        not normalized
        or "/" in normalized
        or "\\" in normalized
        or "\x00" in normalized
        or normalized in {".", ".."}
    ):
        raise ValueError("unsafe_filename")
    suffix = Path(normalized).suffix.casefold()
    if suffix not in _RAW_DOCUMENT_EXTENSIONS:
        raise ValueError("unsupported_file_type")
    return normalized


def _parse_document_upload_multipart(
    content_type: str | None,
    body: bytes,
) -> tuple[str, bytes, str | None, str]:
    """Parse the minimal multipart fields needed by the upload route."""
    if not content_type or "multipart/form-data" not in content_type.lower():
        raise ValueError("validation_failure")

    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    if not message.is_multipart():
        raise ValueError("validation_failure")

    file_name: str | None = None
    file_bytes: bytes | None = None
    title: str | None = None
    conflict_policy = "reject"

    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue
        payload = part.get_payload(decode=True) or b""
        if field_name == "file":
            file_name = part.get_filename()
            file_bytes = payload
        elif field_name == "title":
            decoded_title = payload.decode("utf-8", errors="replace").strip()
            title = decoded_title or None
        elif field_name == "conflict_policy":
            decoded_policy = payload.decode("utf-8", errors="replace").strip()
            if decoded_policy:
                conflict_policy = decoded_policy.lower().replace("-", "_")

    if file_name is None or file_bytes is None:
        raise ValueError("missing_source")
    if conflict_policy not in {"reject", "replace", "create_distinct"}:
        raise ValueError("validation_failure")
    return file_name, file_bytes, title, conflict_policy


def _resolve_upload_destination(
    filename: str,
    conflict_policy: str,
    base_dir: Path = _RAW_DOCUMENT_BASE_DIR,
) -> Path:
    """Resolve destination path according to the explicit upload conflict policy."""
    destination = base_dir / filename
    if not destination.exists():
        return destination

    if conflict_policy == "reject":
        raise FileExistsError("duplicate_conflict")
    if conflict_policy == "replace":
        return destination

    stem = destination.stem
    suffix = destination.suffix
    for index in range(1, 1000):
        candidate = base_dir / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError("duplicate_conflict")


def _build_document_upload_response(
    *,
    destination: Path,
    source_size_bytes: int,
    source_hash: str | None,
    source_status: str,
    title: str | None = None,
    conflict_policy: str | None = None,
    reason: str | None = None,
) -> DocumentUploadResponse:
    """Build a lightweight upload response without preparing the document."""
    sibling_counts = _raw_document_sibling_counts(destination.parent)
    return DocumentUploadResponse(
        doc_name=_raw_document_doc_name(destination, sibling_counts),
        title=title or destination.stem,
        source_filename=destination.name,
        source_type=destination.suffix.casefold().lstrip(".") or "unknown",
        source_size_bytes=source_size_bytes,
        source_hash=source_hash,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        source_status=source_status,
        preparation_status="not_prepared" if source_status == "uploaded" else "unknown",
        conflict_policy=conflict_policy,
        reason=reason,
    )


def _document_upload_failure_source_type(filename: str) -> str:
    """Return a schema-safe source type for failed upload responses."""
    source_type = Path(filename).suffix.casefold().lstrip(".")
    if source_type in {"txt", "pdf"}:
        return source_type
    return "unknown"


def _list_raw_documents(
    query: str | None = None,
    limit: int = 200,
    base_dir: Path | None = None,
) -> list[DocumentListItem]:
    """List raw document files as lightweight discovery candidates."""
    bounded_limit = max(1, min(limit, 200))
    normalized_query = (query or "").strip().casefold()
    base_dir = base_dir or _RAW_DOCUMENT_BASE_DIR
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


def _raw_document_source_type(path: Path) -> str:
    """Return schema-safe source type for supported raw document paths."""
    source_type = path.suffix.casefold().lstrip(".")
    if source_type in {"txt", "pdf"}:
        return source_type
    return "unknown"


def _list_raw_document_management_items(
    query: str | None = None,
    limit: int = 200,
    base_dir: Path | None = None,
) -> list[DocumentManagementItemResponse]:
    """List raw document files for document-management status without heavy payloads."""
    bounded_limit = max(1, min(limit, 200))
    normalized_query = (query or "").strip().casefold()
    base_dir = base_dir or _RAW_DOCUMENT_BASE_DIR
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

    candidates: list[DocumentManagementItemResponse] = []
    for path in raw_paths:
        doc_name = _raw_document_doc_name(path, sibling_counts)
        title = path.stem
        searchable = f"{doc_name} {title}".casefold()
        if normalized_query and normalized_query not in searchable:
            continue
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        candidates.append(
            DocumentManagementItemResponse(
                doc_name=doc_name,
                title=title,
                source=f"raw{path.suffix.casefold()}",
                source_filename=path.name,
                source_type=_raw_document_source_type(path),
                source_size_bytes=stat.st_size,
                source_status="available",
                preparation_status="not_prepared",
                updated_at=modified_at,
            )
        )
        if len(candidates) >= bounded_limit:
            break
    return candidates


def _structured_document_management_item(
    item: DocumentListItem,
) -> DocumentManagementItemResponse:
    """Map a structured document candidate into a lightweight management row."""
    return DocumentManagementItemResponse(
        doc_name=item.doc_name,
        title=item.title,
        source=item.source,
        source_type="unknown",
        source_status="unknown",
        preparation_status="ready",
        review_required=item.review_required,
    )


def _merge_document_management_items(
    structured_items: list[DocumentListItem],
    raw_items: list[DocumentManagementItemResponse],
    limit: int,
) -> list[DocumentManagementItemResponse]:
    """Merge structured readiness with raw source metadata by document name."""
    bounded_limit = max(1, min(limit, 200))
    merged_by_name: dict[str, DocumentManagementItemResponse] = {}

    for item in structured_items:
        merged_by_name[item.doc_name] = _structured_document_management_item(item)

    for raw_item in raw_items:
        existing = merged_by_name.get(raw_item.doc_name)
        if existing is None:
            merged_by_name[raw_item.doc_name] = raw_item
            continue

        existing_sources = {
            source.strip()
            for source in existing.source.split("+")
            if source.strip()
        }
        existing_sources.add(raw_item.source)
        merged_by_name[raw_item.doc_name] = DocumentManagementItemResponse(
            doc_name=existing.doc_name,
            title=existing.title or raw_item.title,
            source="+".join(sorted(existing_sources)),
            source_filename=raw_item.source_filename,
            source_type=raw_item.source_type,
            source_size_bytes=raw_item.source_size_bytes,
            source_status=raw_item.source_status,
            preparation_status=existing.preparation_status,
            review_required=existing.review_required,
            uploaded_at=raw_item.uploaded_at,
            updated_at=raw_item.updated_at,
            last_error=existing.last_error or raw_item.last_error,
        )

    return sorted(
        merged_by_name.values(),
        key=lambda item: item.doc_name.casefold(),
    )[:bounded_limit]


def _validate_document_lifecycle_doc_name(doc_name: str) -> str:
    """Validate a document lifecycle target name for file-backed cleanup."""
    normalized = doc_name.strip()
    if (
        not normalized
        or "/" in normalized
        or "\\" in normalized
        or "\x00" in normalized
        or normalized in {".", ".."}
    ):
        raise ValueError("unsafe_name")
    return normalized


def _resolve_raw_document_delete_path(
    doc_name: str,
    base_dir: Path | None = None,
) -> Path | None:
    """Resolve a raw source path by public doc_name without falling back to content."""
    base_dir = base_dir or _RAW_DOCUMENT_BASE_DIR
    if not base_dir.exists():
        return None

    normalized_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    direct_candidate = base_dir / normalized_doc_name
    if (
        direct_candidate.is_file()
        and direct_candidate.suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS
    ):
        return direct_candidate

    raw_paths = [
        path
        for path in sorted(base_dir.iterdir(), key=lambda candidate: candidate.name.casefold())
        if path.is_file() and path.suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS
    ]
    sibling_counts: dict[str, int] = {}
    for path in raw_paths:
        sibling_counts[path.stem] = sibling_counts.get(path.stem, 0) + 1

    matches = [
        path
        for path in raw_paths
        if _raw_document_doc_name(path, sibling_counts) == normalized_doc_name
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("conflict")
    return matches[0]


def _structured_document_base_dir() -> Path:
    """Return the active file-backed structured document directory."""
    base_dir = getattr(document_artifact_repository, "base_dir", None)
    if isinstance(base_dir, Path):
        return base_dir
    if isinstance(base_dir, str):
        return Path(base_dir)
    return Path("data/structured")


def _resolve_structured_document_delete_path(doc_name: str) -> Path:
    """Resolve the structured JSON artifact path for a lifecycle target."""
    normalized_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    suffix = ".structured.json"
    if normalized_doc_name.endswith(suffix):
        file_name = normalized_doc_name
    elif Path(normalized_doc_name).suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS:
        file_name = f"{Path(normalized_doc_name).stem}{suffix}"
    else:
        file_name = f"{normalized_doc_name}{suffix}"
    return _structured_document_base_dir() / file_name


def _resolve_faiss_document_artifact_dir(doc_name: str) -> Path:
    """Resolve the current file-backed derived artifact directory without creating it."""
    normalized_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    if Path(normalized_doc_name).suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS:
        normalized_doc_name = Path(normalized_doc_name).stem
    return _DERIVED_DOCUMENT_BASE_DIR / normalized_doc_name


def _delete_document_file_backed_artifacts(doc_name: str) -> dict[str, bool]:
    """Delete file-backed document artifacts for an explicit lifecycle request."""
    normalized_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    raw_path = _resolve_raw_document_delete_path(normalized_doc_name)
    derived_dir = _resolve_faiss_document_artifact_dir(normalized_doc_name)

    source_affected = False
    derived_affected = False

    if derived_dir.exists() and not derived_dir.is_dir():
        raise ValueError("storage_failure")

    if raw_path is not None and raw_path.exists():
        raw_path.unlink()
        source_affected = True

    if derived_dir.exists():
        shutil.rmtree(derived_dir)
        derived_affected = True

    return {
        "source_affected": source_affected,
        "derived_resources_affected": derived_affected,
    }


def _document_lifecycle_namespace(doc_name: str) -> str:
    """Return extension-free namespace for structured/derived lifecycle operations."""
    normalized_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    suffix = ".structured.json"
    if normalized_doc_name.endswith(suffix):
        return normalized_doc_name[: -len(suffix)]
    if Path(normalized_doc_name).suffix.casefold() in _RAW_DOCUMENT_EXTENSIONS:
        return Path(normalized_doc_name).stem
    return normalized_doc_name


def _resolve_raw_document_rename_target(
    *,
    current_raw_path: Path | None,
    new_doc_name: str,
) -> Path | None:
    """Resolve raw-source rename target while preserving current extension by default."""
    if current_raw_path is None:
        return None
    normalized_new_name = _validate_document_lifecycle_doc_name(new_doc_name)
    new_path = Path(normalized_new_name)
    if new_path.suffix:
        if new_path.suffix.casefold() not in _RAW_DOCUMENT_EXTENSIONS:
            raise ValueError("unsupported_action")
        target_filename = new_path.name
    else:
        target_filename = f"{normalized_new_name}{current_raw_path.suffix}"
    return current_raw_path.parent / target_filename


def _resolve_structured_document_rename_target(new_doc_name: str) -> Path:
    """Resolve structured JSON rename target for a lifecycle rename request."""
    new_namespace = _document_lifecycle_namespace(new_doc_name)
    return _structured_document_base_dir() / f"{new_namespace}.structured.json"


def _resolve_faiss_document_rename_target(new_doc_name: str) -> Path:
    """Resolve derived artifact directory target for a lifecycle rename request."""
    return _DERIVED_DOCUMENT_BASE_DIR / _document_lifecycle_namespace(new_doc_name)


def _assert_rename_target_available(source: Path | None, target: Path | None) -> None:
    """Fail when a lifecycle rename target already exists for a different artifact."""
    if source is None or target is None:
        return
    try:
        if source.resolve() == target.resolve():
            return
    except FileNotFoundError:
        pass
    if target.exists():
        raise ValueError("conflict")


def _update_renamed_structured_document_metadata(
    structured_path: Path,
    *,
    old_doc_name: str,
    new_doc_name: str,
) -> None:
    """Update lightweight top-level identity metadata after structured file rename."""
    if not structured_path.exists():
        return
    try:
        payload = json.loads(structured_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("storage_failure") from error
    if not isinstance(payload, dict):
        raise ValueError("storage_failure")

    old_namespace = _document_lifecycle_namespace(old_doc_name)
    new_namespace = _document_lifecycle_namespace(new_doc_name)
    if payload.get("chapters") and payload.get("document_task_artifacts"):
        from document_structure.structured_document import StructuredDocument
        from section_tasks.interaction_rename import renamed_interactions
        payload = renamed_interactions(StructuredDocument.from_dict(payload), new_namespace).to_dict()
    if payload.get("document_id") in {None, old_doc_name, old_namespace}:
        payload["document_id"] = new_namespace
    if payload.get("title") in {None, old_doc_name, old_namespace}:
        payload["title"] = new_namespace
    try:
        structured_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError("storage_failure") from error


def _rename_document_file_backed_artifacts(
    *,
    doc_name: str,
    new_doc_name: str,
) -> dict[str, bool | str]:
    """Rename file-backed document artifacts for an explicit lifecycle request."""
    old_doc_name = _validate_document_lifecycle_doc_name(doc_name)
    normalized_new_doc_name = _validate_document_lifecycle_doc_name(new_doc_name)
    if _document_lifecycle_namespace(old_doc_name) == _document_lifecycle_namespace(
        normalized_new_doc_name
    ):
        raise ValueError("conflict")

    raw_path = _resolve_raw_document_delete_path(old_doc_name)
    raw_target = _resolve_raw_document_rename_target(
        current_raw_path=raw_path,
        new_doc_name=normalized_new_doc_name,
    )
    structured_path = _resolve_structured_document_delete_path(old_doc_name)
    structured_target = _resolve_structured_document_rename_target(normalized_new_doc_name)
    derived_dir = _resolve_faiss_document_artifact_dir(old_doc_name)
    derived_target = _resolve_faiss_document_rename_target(normalized_new_doc_name)

    source_exists = raw_path is not None and raw_path.exists()
    structured_exists = structured_path.exists()
    derived_exists = derived_dir.exists()
    if not (source_exists or structured_exists or derived_exists):
        raise FileNotFoundError("missing_document")

    _assert_rename_target_available(raw_path if source_exists else None, raw_target)
    _assert_rename_target_available(
        structured_path if structured_exists else None,
        structured_target,
    )
    _assert_rename_target_available(derived_dir if derived_exists else None, derived_target)

    source_affected = False
    structured_affected = False
    derived_affected = False

    if source_exists and raw_path is not None and raw_target is not None:
        raw_path.rename(raw_target)
        source_affected = True

    if structured_exists:
        structured_path.rename(structured_target)
        _update_renamed_structured_document_metadata(
            structured_target,
            old_doc_name=old_doc_name,
            new_doc_name=normalized_new_doc_name,
        )
        structured_affected = True

    if derived_exists:
        if not derived_dir.is_dir():
            raise ValueError("storage_failure")
        derived_dir.rename(derived_target)
        derived_affected = True

    return {
        "new_doc_name": _document_lifecycle_namespace(normalized_new_doc_name),
        "source_affected": source_affected,
        "structured_artifacts_affected": structured_affected,
        "derived_resources_affected": derived_affected,
    }


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
                interaction_availability=(
                    None
                    if getattr(content_block, "interaction_availability", None) is None
                    else dict(content_block.interaction_availability)
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

    prefix = "prepare_structured_document_llm_enhanced_rejected:"
    if lowered.startswith(prefix):
        reason = lowered[len(prefix):]
        if reason in {"enhancement_prompt_budget_exceeded", "local_resource_limit"}:
            return 413
        if reason in {"enhancement_budget_unavailable", "enhancement_output_budget_invalid",
                      "enhancement_provider_unavailable"}:
            return 503
        if reason in {"enhancement_provider_failed", "enhancement_provider_transient"}:
            return 502
        if reason == "enhancement_configuration_changed":
            return 409
        if reason == "enhancement_candidate_rejected:unexpected_validation_error":
            return 500
        if reason == "enhancement_candidate_rejected" or reason.startswith("enhancement_candidate_rejected:"):
            return 422
        return 500

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


from api_schemas import NoteReadRequest, NoteSaveRequest, NoteDeleteRequest, NoteResponse
from api_schemas import (SummaryInteractionReadRequest, SummaryInteractionGenerateRequest,
                         SummaryInteractionRefreshRequest, SummaryInteractionResponse)


def _summary_route(operation, request, response):
    try:
        result = getattr(section_task_coordinator, operation)(**_reading_target_kwargs(request.target))
        mapped = SummaryInteractionResponse(envelope=_map_reading_interaction_envelope(result),
                                             payload=result.payload if result.status == "completed" else None)
        response.status_code = _reading_interaction_http_status(result.status)
        return mapped
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post("/documents/reading-interactions/summary/read", response_model=SummaryInteractionResponse)
def read_summary_interaction(request: SummaryInteractionReadRequest, response: Response):
    return _summary_route("read_summary_artifact", request, response)


@app.post("/documents/reading-interactions/summary/generate", response_model=SummaryInteractionResponse)
def generate_summary_interaction(request: SummaryInteractionGenerateRequest, response: Response):
    return _summary_route("generate_summary_artifact", request, response)


@app.post("/documents/reading-interactions/summary/refresh", response_model=SummaryInteractionResponse)
def refresh_summary_interaction(request: SummaryInteractionRefreshRequest, response: Response):
    return _summary_route("refresh_summary_artifact", request, response)


@app.post("/documents/reading-interactions/note/read", response_model=NoteResponse)
def read_note_interaction(request: NoteReadRequest):
    try:
        return section_task_coordinator.read_note(**_reading_target_kwargs(request.target))
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post("/documents/reading-interactions/note/save", response_model=NoteResponse)
def save_note_interaction(request: NoteSaveRequest):
    try:
        return section_task_coordinator.save_note(
            **_reading_target_kwargs(request.target), text=request.text,
            expected_revision=request.expected_revision, mutation_id=request.mutation_id,
            target_token=request.target_token)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


@app.post("/documents/reading-interactions/note/delete", response_model=NoteResponse)
def delete_note_interaction(request: NoteDeleteRequest):
    try:
        return section_task_coordinator.delete_note(
            **_reading_target_kwargs(request.target),
            expected_revision=request.expected_revision, target_token=request.target_token)
    except Exception as error:
        _raise_reading_interaction_http_exception(error)


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
        "content_block_id": target.content_block_id,
        "source_hash": target.source_hash,
        "source_structure_version": target.source_structure_version,
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
            content_block_id=target.content_block_id,
            title=target.title,
        ),
        interaction_type=result.interaction_type,
        status=result.status,
        artifact_id=result.artifact_id,
        session_id=result.session_id,
        generated_at=metadata.get("generated_at"),
        updated_at=metadata.get("updated_at"),
        source_structure_version=metadata.get("source_structure_version"),
        source_hash=metadata.get("source_hash"),
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


@app.get("/documents/management", response_model=DocumentManagementListResponse)
def list_document_management_items(
    q: str | None = Query(
        None,
        description="Optional case-insensitive substring query against document name/title.",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Maximum number of document management rows to return.",
    ),
):
    """List lightweight document source/preparation status for file management UI."""
    normalized_query = None if q is None or not q.strip() else q.strip()
    try:
        structured_items = document_artifact_repository.list_documents(
            query=normalized_query,
            limit=200,
        )
        raw_items = _list_raw_document_management_items(
            query=normalized_query,
            limit=200,
        )
        merged_items = _merge_document_management_items(
            structured_items=structured_items,
            raw_items=raw_items,
            limit=limit,
        )
        return DocumentManagementListResponse(
            items=merged_items,
            query=normalized_query,
            total=len(merged_items),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, response: Response):
    """Store a raw document source candidate without preparing or parsing it."""
    with log_stage(logger, "upload_receive") as timing:
        body = await request.body()
        timing["body_bytes"] = len(body)
    try:
        with log_stage(logger, "upload_multipart_parse"):
            filename, file_bytes, title, conflict_policy = _parse_document_upload_multipart(
                request.headers.get("content-type"),
                body,
            )
        normalized_filename = _normalize_upload_filename(filename)
        logger.info("upload_metadata filename=%s file_bytes=%s conflict_policy=%s",
                    json.dumps(normalized_filename, ensure_ascii=False), len(file_bytes), conflict_policy)

        if not file_bytes:
            logger.info("upload_rejected reason=missing_source")
            response.status_code = 400
            return DocumentUploadResponse(
                doc_name=Path(normalized_filename).stem,
                title=title or Path(normalized_filename).stem,
                source_filename=normalized_filename,
                source_type=_document_upload_failure_source_type(normalized_filename),
                source_size_bytes=0,
                source_status="failed",
                preparation_status="unknown",
                conflict_policy=conflict_policy,
                reason="missing_source",
            )
        if len(file_bytes) > _DOCUMENT_UPLOAD_MAX_BYTES:
            logger.info("upload_rejected reason=file_too_large max_bytes=%s", _DOCUMENT_UPLOAD_MAX_BYTES)
            response.status_code = 413
            return DocumentUploadResponse(
                doc_name=Path(normalized_filename).stem,
                title=title or Path(normalized_filename).stem,
                source_filename=normalized_filename,
                source_type=_document_upload_failure_source_type(normalized_filename),
                source_size_bytes=len(file_bytes),
                source_status="failed",
                preparation_status="unknown",
                conflict_policy=conflict_policy,
                reason="file_too_large",
            )

        _RAW_DOCUMENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        destination = _resolve_upload_destination(
            normalized_filename,
            conflict_policy,
            _RAW_DOCUMENT_BASE_DIR,
        )
        with log_stage(logger, "upload_write", filename=destination.name, file_bytes=len(file_bytes)):
            destination.write_bytes(file_bytes)
        with log_stage(logger, "upload_hash", file_bytes=len(file_bytes)):
            source_hash = hashlib.sha256(file_bytes).hexdigest()
        response.status_code = 201
        logger.info("upload_stored doc_name=%s filename=%s file_bytes=%s preparation_started=false",
                    json.dumps(destination.stem, ensure_ascii=False),
                    json.dumps(destination.name, ensure_ascii=False), len(file_bytes))
        return _build_document_upload_response(
            destination=destination,
            source_size_bytes=len(file_bytes),
            source_hash=source_hash,
            source_status="uploaded",
            title=title,
            conflict_policy=conflict_policy,
        )
    except FileExistsError:
        logger.info("upload_rejected reason=duplicate_conflict")
        response.status_code = 409
        filename = filename if "filename" in locals() else "upload"
        normalized_filename = Path(filename).name
        return DocumentUploadResponse(
            doc_name=Path(normalized_filename).stem or "upload",
            title=title if "title" in locals() else None,
            source_filename=normalized_filename,
            source_type=_document_upload_failure_source_type(normalized_filename),
            source_size_bytes=len(file_bytes) if "file_bytes" in locals() else 0,
            source_status="failed",
            preparation_status="unknown",
            conflict_policy=conflict_policy if "conflict_policy" in locals() else "reject",
            reason="duplicate_conflict",
        )
    except ValueError as error:
        reason = str(error)
        status_code = 415 if reason == "unsupported_file_type" else 400
        response.status_code = status_code
        filename = filename if "filename" in locals() else "upload"
        normalized_filename = Path(filename).name or "upload"
        source_type = _document_upload_failure_source_type(normalized_filename)
        if reason not in {
            "unsupported_file_type",
            "unsafe_filename",
            "missing_source",
            "validation_failure",
        }:
            reason = "validation_failure"
        logger.info("upload_rejected reason=%s", reason)
        return DocumentUploadResponse(
            doc_name=Path(normalized_filename).stem or "upload",
            title=title if "title" in locals() else None,
            source_filename=normalized_filename,
            source_type=source_type,
            source_size_bytes=len(file_bytes) if "file_bytes" in locals() else 0,
            source_status="failed",
            preparation_status="unknown",
            conflict_policy=conflict_policy if "conflict_policy" in locals() else "reject",
            reason=reason,
        )
    except Exception:
        logger.exception("document_upload_failed")
        response.status_code = 500
        filename = filename if "filename" in locals() else "upload"
        normalized_filename = Path(filename).name or "upload"
        return DocumentUploadResponse(
            doc_name=Path(normalized_filename).stem or "upload",
            title=title if "title" in locals() else None,
            source_filename=normalized_filename,
            source_type=_document_upload_failure_source_type(normalized_filename),
            source_size_bytes=len(file_bytes) if "file_bytes" in locals() else 0,
            source_status="failed",
            preparation_status="unknown",
            conflict_policy=conflict_policy if "conflict_policy" in locals() else "reject",
            reason="storage_failure",
        )


@app.post(
    "/documents/lifecycle/delete",
    response_model=DocumentLifecycleActionResponse,
)
def delete_document_lifecycle(
    request: DocumentLifecycleActionRequest,
    response: Response,
):
    """Explicitly delete source, derived files, and active structured storage data."""
    try:
        doc_name = _validate_document_lifecycle_doc_name(request.doc_name)
        if request.action != "delete":
            response.status_code = 400
            return DocumentLifecycleActionResponse(
                doc_name=doc_name,
                action=request.action,
                status="rejected",
                reason="unsupported_action",
                deleted=False,
            )

        if not request.confirm:
            response.status_code = 400
            return DocumentLifecycleActionResponse(
                doc_name=doc_name,
                action="delete",
                status="rejected",
                reason="validation_failure",
                deleted=False,
            )

        effects = _delete_document_file_backed_artifacts(doc_name)
        # Keep the discoverable structured record until file cleanup succeeds.
        effects["structured_artifacts_affected"] = (
            document_artifact_repository.delete_document(doc_name)
        )
        deleted = any(effects.values())
        if not deleted:
            response.status_code = 404
            return DocumentLifecycleActionResponse(
                doc_name=doc_name,
                action="delete",
                status="rejected",
                reason="missing_document",
                deleted=False,
            )

        return DocumentLifecycleActionResponse(
            doc_name=doc_name,
            action="delete",
            status="accepted",
            source_affected=effects["source_affected"],
            structured_artifacts_affected=effects["structured_artifacts_affected"],
            derived_resources_affected=effects["derived_resources_affected"],
            archived=False,
            deleted=True,
        )
    except ValueError as error:
        reason = str(error)
        if reason not in {
            "unsafe_name",
            "conflict",
            "validation_failure",
            "storage_failure",
        }:
            reason = "validation_failure"
        response.status_code = 409 if reason == "conflict" else 400
        return DocumentLifecycleActionResponse(
            doc_name=(request.doc_name or "document").strip() or "document",
            action="delete",
            status="conflict" if reason == "conflict" else "rejected",
            reason=reason if reason != "unsafe_name" else "unsafe_name",
            deleted=False,
        )
    except Exception:
        logger.exception("document_lifecycle_delete_failed")
        response.status_code = 500
        return DocumentLifecycleActionResponse(
            doc_name=request.doc_name,
            action="delete",
            status="failed",
            reason="storage_failure",
            deleted=False,
        )


@app.post(
    "/documents/lifecycle/rename",
    response_model=DocumentLifecycleActionResponse,
)
def rename_document_lifecycle(
    request: DocumentLifecycleActionRequest,
    response: Response,
):
    """Explicitly rename a document candidate and file-backed derived artifacts."""
    try:
        doc_name = _validate_document_lifecycle_doc_name(request.doc_name)
        if request.action != "rename":
            response.status_code = 400
            return DocumentLifecycleActionResponse(
                doc_name=doc_name,
                action=request.action,
                status="rejected",
                reason="unsupported_action",
            )
        if request.new_doc_name is None:
            response.status_code = 400
            return DocumentLifecycleActionResponse(
                doc_name=doc_name,
                action="rename",
                status="rejected",
                reason="validation_failure",
            )

        effects = _rename_document_file_backed_artifacts(
            doc_name=doc_name,
            new_doc_name=request.new_doc_name,
        )
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        if isinstance(document_artifact_repository, PostgresStructuredDocumentArtifactRepository):
            from section_tasks.interaction_rename import renamed_interactions
            try:
                effects["structured_artifacts_affected"] = document_artifact_repository.rename_document(
                    _document_lifecycle_namespace(doc_name),
                    _document_lifecycle_namespace(request.new_doc_name), renamed_interactions)
            except Exception:
                _rename_document_file_backed_artifacts(doc_name=request.new_doc_name, new_doc_name=doc_name)
                raise
        return DocumentLifecycleActionResponse(
            doc_name=doc_name,
            action="rename",
            status="accepted",
            new_doc_name=str(effects["new_doc_name"]),
            source_affected=bool(effects["source_affected"]),
            structured_artifacts_affected=bool(
                effects["structured_artifacts_affected"]
            ),
            derived_resources_affected=bool(effects["derived_resources_affected"]),
        )
    except FileNotFoundError:
        response.status_code = 404
        return DocumentLifecycleActionResponse(
            doc_name=request.doc_name,
            action="rename",
            status="rejected",
            reason="missing_document",
        )
    except ValueError as error:
        reason = str(error)
        if reason not in {
            "unsafe_name",
            "unsupported_action",
            "conflict",
            "validation_failure",
            "storage_failure",
        }:
            reason = "validation_failure"
        response.status_code = 409 if reason == "conflict" else 400
        return DocumentLifecycleActionResponse(
            doc_name=(request.doc_name or "document").strip() or "document",
            action="rename",
            status="conflict" if reason == "conflict" else "rejected",
            reason=reason,
        )
    except Exception:
        logger.exception("document_lifecycle_rename_failed")
        response.status_code = 500
        return DocumentLifecycleActionResponse(
            doc_name=request.doc_name,
            action="rename",
            status="failed",
            reason="storage_failure",
        )


_PDF_EXTRACTION_FAILURE_HTTP_STATUS = {
    PdfExtractionFailureReason.MODE_UNAVAILABLE: 501,
    PdfExtractionFailureReason.MODE_NOT_APPLICABLE: 422,
    PdfExtractionFailureReason.NO_TEXT: 422,
    PdfExtractionFailureReason.ENGINE_UNAVAILABLE: 503,
    PdfExtractionFailureReason.QUALITY_FAILED: 422,
    PdfExtractionFailureReason.BUDGET_EXCEEDED: 422,
    PdfExtractionFailureReason.EXTRACTION_FAILED: 502,
}
_PREPARATION_ERROR_RESPONSES = {
    422: {"model": PdfExtractionFailureResponse | PreparationValidationFailureResponse},
    501: {"model": PdfExtractionFailureResponse},
    502: {"model": PdfExtractionFailureResponse},
    503: {"model": PdfExtractionFailureResponse},
}


def _pdf_extraction_failure_response(request, error: PdfExtractionError) -> JSONResponse:
    payload = PdfExtractionFailureResponse(
        doc_name=request.doc_name,
        reason=error.reason,
        detail=str(error),
        pdf_extraction=PdfExtractionMetadataResponse(requested_mode=request.pdf_extraction_mode),
    )
    return JSONResponse(
        status_code=_PDF_EXTRACTION_FAILURE_HTTP_STATUS[error.reason],
        content=payload.model_dump(mode="json"),
    )


def _pdf_extraction_metadata(request, assets) -> PdfExtractionMetadataResponse:
    return PdfExtractionMetadataResponse(
        requested_mode=request.pdf_extraction_mode,
        effective_mode=assets.effective_pdf_extraction_mode,
        provenance=assets.pdf_extraction_provenance,
    )


@app.post("/documents/prepare", response_model=PrepareDocumentResponse, responses=_PREPARATION_ERROR_RESPONSES)
def prepare_document(request: PrepareDocumentRequest, response: Response):
    """Prepare document artifacts through REST for deterministic parser-mode testing."""
    try:
        assets = qa_coordinator.document_preparation_pipeline.prepare(
            doc_name=request.doc_name,
            force_rebuild=request.force_rebuild,
            mode=request.mode,
            structured_parser_mode=request.structured_parser_mode,
            pdf_extraction_mode=request.pdf_extraction_mode,
            toc_hint=request.toc_hint,
            enhancement_intent=request.enhancement_intent,
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
            pdf_extraction=_pdf_extraction_metadata(request, assets),
        )
    except PdfExtractionError as error:
        return _pdf_extraction_failure_response(request, error)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/prepare-task-layout", response_model=PrepareTaskLayoutResponse,
          responses={**_PREPARATION_ERROR_RESPONSES, 409: {"description": "Persisted task layout is not ready; explicit preparation guidance in detail."}})
def prepare_task_layout(request: PrepareTaskLayoutRequest):
    """Prepare a document when needed, then return its task-layout projection."""
    try:
        assets = None
        if request.layout_only:
            if request.toc_hint != "auto":
                raise HTTPException(status_code=400, detail="layout_only cannot apply toc_hint")
            if request.force_rebuild:
                raise HTTPException(status_code=400, detail="layout_only cannot rebuild source structure")
            if request.include_anchor_page_evidence:
                raise HTTPException(status_code=400, detail="layout_only cannot load raw page evidence; request evidence separately")
        else:
            assets = qa_coordinator.document_preparation_pipeline.prepare(
                doc_name=request.doc_name,
                force_rebuild=request.force_rebuild,
                mode="base",
                structured_parser_mode=request.structured_parser_mode,
                pdf_extraction_mode=request.pdf_extraction_mode,
                toc_hint=request.toc_hint,
                enhancement_intent=request.enhancement_intent,
            )
            if not assets.structured_document_ready:
                detail = " | ".join(assets.errors) or "structured document preparation failed"
                raise HTTPException(status_code=400, detail=detail)

        section_task_coordinator.ensure_document_task_layout(
            doc_name=request.doc_name,
            refresh_task_units=request.refresh_task_units,
            task_unit_split_mode=request.task_unit_split_mode,
            semantic_top_k_candidates=request.semantic_top_k_candidates,
        )
        layout = get_document_task_layout(
            GetDocumentTaskLayoutRequest(
                doc_name=request.doc_name,
                refresh_task_units=False,
                task_unit_split_mode=request.task_unit_split_mode,
                semantic_top_k_candidates=request.semantic_top_k_candidates,
                include_anchor_page_evidence=request.include_anchor_page_evidence,
            )
        )
        return PrepareTaskLayoutResponse(
            **layout.model_dump(),
            pdf_extraction=(
                _pdf_extraction_metadata(request, assets) if assets is not None
                else PdfExtractionMetadataResponse(requested_mode=request.pdf_extraction_mode)
            ),
        )
    except TaskLayoutNotReadyError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail=str(TaskLayoutNotReadyError("missing_structure")))
    except PdfExtractionError as error:
        return _pdf_extraction_failure_response(request, error)
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


@app.post("/documents/task-layout", response_model=DocumentTaskLayoutResponse,
          responses={409: {"description": "Task layout requires explicit preparation; detail contains recovery guidance."}})
def get_document_task_layout(request: GetDocumentTaskLayoutRequest):
    """Read current effective document-task layout snapshot."""
    try:
        with log_stage(logger, "task_layout_projection", doc_name=request.doc_name,
                       refresh_task_units=request.refresh_task_units,
                       split_mode=request.task_unit_split_mode):
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
                title_origin=section.title_origin,
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
                    toc_hint=layout.parse_provenance.toc_hint,
                    toc_observation=layout.parse_provenance.toc_observation,
                    enhancement_intent=layout.parse_provenance.enhancement_intent,
                    implicit_single_body=layout.parse_provenance.implicit_single_body,
                    declaration_unconfirmed=layout.parse_provenance.declaration_unconfirmed,
                )
            ),
        )
    except TaskLayoutNotReadyError as error:
        raise HTTPException(status_code=409, detail=str(error))
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


@app.post("/documents/reparse-structure/preflight", response_model=EnhancementPreflightResponse)
def preflight_document_structure(request: EnhancementPreflightRequest):
    return enhancement_preflight_service.preflight(
        doc_name=request.doc_name, toc_hint=request.toc_hint,
        enhancement_intent=request.enhancement_intent,
    )


@app.post("/documents/reparse-structure/result", response_model=EnhancementOperationResponse)
def enhancement_operation_result(request: EnhancementOperationRequest):
    """Read process-local operation evidence; never infer success from a version."""
    return enhancement_preflight_service.operations.lookup(request.model_dump())


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
        def execute():
            return section_task_coordinator.reparse_document_structure(
                doc_name=request.doc_name,
                parser_mode=normalized_parser_mode,
                toc_hint=request.toc_hint,
                enhancement_intent=request.enhancement_intent,
                **({"discard_interactions": request.discard_interactions,
                    "confirmation_token": request.confirmation_token}
                   if normalized_parser_mode == "llm_enhanced" else {}),
            )
        result = (enhancement_preflight_service.operations.execute(request.model_dump(), execute)
                  if normalized_parser_mode == "llm_enhanced" else execute())
        if result.success:
            return ReparseDocumentStructureResponse(
                success=True,
                doc_name=result.doc_name,
                parser_mode=result.parser_mode,
                structured_document_path=result.structured_document_path,
                error=None,
                section_count=result.section_count,
                structure_version=getattr(result, "structure_version", None),
                change_kind=getattr(result, "change_kind", None),
                committed=getattr(result, "committed", None),
                interactions_cleared=getattr(result, "interactions_cleared", None),
                reading_layout_required=getattr(result, "reading_layout_required", None),
            )

        response.status_code = _resolve_reparse_failure_status(result.error or "")
        return ReparseDocumentStructureResponse(
            success=False,
            doc_name=result.doc_name,
            parser_mode=result.parser_mode,
            structured_document_path=result.structured_document_path,
            error=result.error,
            section_count=result.section_count,
            retryable=getattr(result, "retryable", False),
        )
    except EnhancementConflict as error:
        if error.capacity is not None:
            from api_schemas import EnhancementCapacityResponse
            return JSONResponse(status_code=error.status_code, content={
                "detail": error.reason,
                "capacity": EnhancementCapacityResponse.model_validate(error.capacity).model_dump()})
        raise HTTPException(status_code=error.status_code, detail=error.reason)
    except HTTPException:
        raise
    except ValueError as error:
        if normalized_parser_mode == "llm_enhanced":
            raise HTTPException(status_code=500, detail="enhancement_execution_failed") from None
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        if normalized_parser_mode == "llm_enhanced":
            raise HTTPException(status_code=500, detail="enhancement_execution_failed") from None
        raise HTTPException(status_code=500, detail=str(error))
