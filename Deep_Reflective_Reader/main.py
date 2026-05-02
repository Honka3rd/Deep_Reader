from uuid import uuid4
import re

from fastapi import FastAPI, HTTPException, Response

from app.qa_coordinator import QACoordinator
from api_schemas import (
    ArtifactAvailabilityResponse,
    DocumentTaskLayoutChapterResponse,
    ChapterQuizRequest,
    ChapterQuizResponse,
    PrepareDocumentRequest,
    PrepareDocumentResponse,
    AskDocumentRequest,
    AskDocumentResponse,
    DocumentTaskLayoutResponse,
    EnhancedParseRecommendationResponse,
    GetDocumentTaskLayoutRequest,
    QuizQuestionResponse,
    ReparseDocumentStructureRequest,
    ReparseDocumentStructureResponse,
    SectionTaskLayoutResponse,
    SectionQuizResponse,
    SectionTaskRequest,
    SectionTaskResponse,
    SummarizeChapterRequest,
    SummarizeChapterResponse,
    TaskUnitMetadataResponse,
    StatusResponse,
)

# 建立 FastAPI app
app = FastAPI(
    title="Deep Reader API",
    version="0.1.0",
)

# ⭐ QA coordinator 是自由問答主線 singleton
qa_coordinator = QACoordinator()
section_task_coordinator = qa_coordinator.container.section_task_coordinator()


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
            )

        chapters = [
            DocumentTaskLayoutChapterResponse(
                chapter_id=chapter.chapter_id,
                title=chapter.title,
                level=chapter.level,
                chapter_role=chapter.chapter_role,
                sections=[_section_response(section) for section in chapter.sections],
                artifacts=_artifact_response(chapter.artifacts),
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
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/documents/reparse-structure", response_model=ReparseDocumentStructureResponse)
def reparse_document_structure(
    request: ReparseDocumentStructureRequest,
    response: Response,
):
    """Run explicit structure reparse action and replace single active source on success."""
    normalized_parser_mode = request.parser_mode.strip().lower().replace("-", "_")
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
