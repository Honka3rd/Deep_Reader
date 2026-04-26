from uuid import uuid4
import re

from fastapi import FastAPI, HTTPException, Response

from app.qa_coordinator import QACoordinator
from api_schemas import (
    PrepareDocumentRequest,
    PrepareDocumentResponse,
    AskDocumentRequest,
    AskDocumentResponse,
    QuizQuestionResponse,
    SectionQuizResponse,
    SectionTaskRequest,
    SectionTaskResponse,
    SummarizeChapterRequest,
    SummarizeChapterResponse,
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
        )
        if result.success:
            return SectionTaskResponse(
                doc_name=request.doc_name,
                section_id=request.section_id,
                success=True,
                result=result.payload,
                reason=None,
            )
        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SectionTaskResponse(
            doc_name=request.doc_name,
            section_id=request.section_id,
            success=False,
            result=None,
            reason=result.reason,
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
            )
        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SectionQuizResponse(
            doc_name=request.doc_name,
            section_id=request.section_id,
            success=False,
            questions=None,
            reason=result.reason,
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
    """Run summary task for one chapter resolved by exact title."""
    try:
        result = section_task_coordinator.summarize_chapter(
            doc_name=request.doc_name,
            chapter_title=request.chapter_title,
        )
        if result.success:
            return SummarizeChapterResponse(
                doc_name=request.doc_name,
                chapter_title=request.chapter_title,
                success=True,
                result=result.payload,
                reason=None,
            )

        response.status_code = _resolve_section_task_failure_status(result.reason)
        return SummarizeChapterResponse(
            doc_name=request.doc_name,
            chapter_title=request.chapter_title,
            success=False,
            result=None,
            reason=result.reason,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
