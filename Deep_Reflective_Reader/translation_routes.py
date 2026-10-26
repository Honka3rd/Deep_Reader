"""Minimal transport for saved block translation reads and explicit generation."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from api_schemas import (
    ReadingInteractionTargetResponse, TranslationInteractionEnvelope, TranslationInteractionErrorResponse,
    TranslationInteractionPayload, TranslationInteractionRequest, TranslationInteractionResponse,
    TranslationReadRequest, TranslationReadResponse,
)
from app.translation_action import TranslationAction
from section_tasks.translation_contracts import TranslationError
from section_tasks.translation_prompt import TRANSLATION_PROMPT_VERSION


logger = logging.getLogger(__name__)
_ERROR_STATUS = {
    "translation_document_missing": 404, "translation_target_missing": 404,
    "translation_target_invalid": 422, "translation_request_invalid": 422,
    "translation_invalid_source_language": 422, "translation_invalid_target_language": 422,
    "translation_source_language_required": 422, "translation_invalid_text": 422,
    "translation_same_language": 422,
    "translation_source_changed": 409, "translation_capacity_exceeded": 413,
    "translation_provider_failed": 502, "translation_invalid_output": 502,
    "translation_capacity_unavailable": 503, "translation_profile_unavailable": 503,
    "translation_service_unavailable": 503,
    "translation_mutation_conflict": 409, "translation_history_limit": 409,
}


class _TranslationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                response = await handler(request)
            except RequestValidationError:
                response = JSONResponse({"detail": "translation_request_invalid"}, status_code=422)
            except FileNotFoundError:
                response = JSONResponse({"detail": "translation_document_missing"}, status_code=404)
            except TranslationError as error:
                reason = error.code if error.code in _ERROR_STATUS else "translation_service_unavailable"
                response = JSONResponse({"detail": reason}, status_code=_ERROR_STATUS[reason])
            except Exception:
                logger.warning("translation_action_failed reason=translation_service_unavailable")
                response = JSONResponse({"detail": "translation_service_unavailable"}, status_code=503)
            response.headers["Cache-Control"] = "no-store"
            return response

        return guarded


router = APIRouter(route_class=_TranslationRoute, tags=["reading-interactions"])


def get_translation_action(request: Request) -> TranslationAction:
    service = getattr(request.app.state, "translation_action", None)
    if service is None:
        raise TranslationError("translation_service_unavailable")
    return service


def _target(doc_name, target):
    return ReadingInteractionTargetResponse(
        doc_name=doc_name, document_id=target.document_id, document_title=target.document_title,
        target_type=target.target_level, target_id=target.target_id, chapter_id=target.chapter_id,
        section_id=target.section_id, task_unit_id=target.task_unit_id, content_block_id=target.content_block_id)


@router.post("/documents/reading-interactions/translate/read", response_model=TranslationReadResponse)
def read_translations(request: TranslationReadRequest,
                      service: Annotated[TranslationAction, Depends(get_translation_action)]):
    result = service.read(doc_name=request.doc_name, content_block_id=request.content_block_id)
    return {**result, "target": _target(request.doc_name, result["target"])}


@router.post("/documents/reading-interactions/translate/generate",
             response_model=TranslationInteractionResponse,
             responses={code: {"model": TranslationInteractionErrorResponse} for code in (404, 409, 413, 422, 502, 503)})
def generate_translation(request: TranslationInteractionRequest,
                         service: Annotated[TranslationAction, Depends(get_translation_action)]):
    result = service.generate(doc_name=request.doc_name, content_block_id=request.content_block_id,
                              target_language=request.target_language,
                              source_language=request.source_language, mutation_id=request.mutation_id)
    target = result.target
    return TranslationInteractionResponse(
        envelope=TranslationInteractionEnvelope(
            target=_target(request.doc_name, target), persisted=result.record is not None,
            status=result.translation.status, source_structure_version=target.source_structure_version,
            reading_revision=target.reading_revision, effective_content_hash=result.effective_content_hash,
            prompt_instruction_version=TRANSLATION_PROMPT_VERSION),
        payload=TranslationInteractionPayload(source_language=result.translation.source_language,
            target_language=result.translation.target_language, translated_text=result.translation.translated_text),
        record=result.record)
