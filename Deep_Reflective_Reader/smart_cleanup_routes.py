"""Transport-only standalone cleanup routes, owned by main.py's API module."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from api_document_name import DocumentName
from api_schemas import (
    SMART_CLEANUP_REASONS, SmartCleanupErrorResponse, SmartCleanupExecuteRequest, SmartCleanupOperationId,
    SmartCleanupOperationResponse, SmartCleanupPreflightRequest, SmartCleanupPreflightResponse,
)
from app.standalone_cleanup import CleanupUnavailable, StandaloneCleanupService


LOGGER = logging.getLogger(__name__)


def _status(reason, *, lookup=False):
    if reason == "cleanup_document_missing":
        return 404
    if reason in {"cleanup_service_unavailable", "cleanup_preflight_unavailable", "capacity_unavailable",
                  "cleanup_reconciliation_unavailable", "cleanup_execution_unavailable", "cleanup_operation_capacity"}:
        return 503
    if reason in {"cleanup_identity_changed", "cleanup_in_progress", "cleanup_confirmation_required",
                  "cleanup_confirmation_invalid", "cleanup_confirmation_expired", "source_changed",
                  "reading_revision_changed", "cleanup_policy_changed"}:
        return 409
    if reason == "cleanup_outcome_unknown" and not lookup:
        return 409
    return 200


class _CleanupRoute(APIRoute):
    """Keep token/body validation failures and private exceptions off the wire."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                if request.query_params:
                    raise HTTPException(422, "cleanup_request_invalid")
                if request.method == "GET" and await request.body():
                    raise HTTPException(422, "cleanup_request_invalid")
                response = await handler(request)
            except RequestValidationError:
                response = JSONResponse({"detail": "cleanup_request_invalid"}, status_code=422)
            except FileNotFoundError:
                response = JSONResponse({"detail": "cleanup_document_missing"}, status_code=404)
            except CleanupUnavailable as error:
                reason = str(error) if str(error) in SMART_CLEANUP_REASONS else "cleanup_service_unavailable"
                response = JSONResponse({"detail": reason}, status_code=_status(reason))
            except HTTPException as error:
                safe = isinstance(error.detail, str) and error.detail in SMART_CLEANUP_REASONS | {"cleanup_request_invalid"}
                response = JSONResponse({"detail": error.detail if safe else "cleanup_service_unavailable"},
                                        status_code=error.status_code if safe else 503)
            except Exception:
                LOGGER.warning("smart_cleanup_transport_failed")
                response = JSONResponse({"detail": "cleanup_service_unavailable"}, status_code=503)
            response.headers["Cache-Control"] = "no-store"
            return response

        return guarded


router = APIRouter(route_class=_CleanupRoute, tags=["smart-cleanup"])


def get_smart_cleanup_service(request: Request) -> StandaloneCleanupService | None:
    # The configured service owns the namespace, just like the local repositories.
    # SCLEAN-07 supplies runtime ports. Never infer namespace from client headers/body.
    return getattr(request.app.state, "smart_cleanup_service", None)


Service = Annotated[StandaloneCleanupService | None, Depends(get_smart_cleanup_service)]


def _project(result):
    state = result.state
    commit = state.commit if state else "unknown"
    return SmartCleanupOperationResponse(operation_id=result.operation_id, phase=result.phase,
        commit=commit, outcome=state.outcome if state else None, reason=result.reason,
        retry_action=result.retry, durability=result.durability, pending_review_count=result.pending_review_count,
        reading_ready=True if result.phase == "finished" and commit == "committed" else (
            False if result.phase == "layout" else None))


def _responses(model):
    return {code: {"model": SmartCleanupErrorResponse if code == 422 else model | SmartCleanupErrorResponse}
            for code in (404, 409, 422, 503)}


@router.post("/documents/{doc_name}/smart-cleanup/preflight", response_model=SmartCleanupPreflightResponse,
             responses=_responses(SmartCleanupPreflightResponse))
def preflight(doc_name: DocumentName, response: Response, service: Service,
              body: SmartCleanupPreflightRequest = Body(default=SmartCleanupPreflightRequest())):
    if service is None:
        response.status_code = 503
        return SmartCleanupPreflightResponse(execution_available=False, reason="cleanup_service_unavailable")
    result = service.preflight(doc_name)
    projected = SmartCleanupPreflightResponse(execution_available=result.ready, reason=result.reason,
        operation_id=result.operation_id, confirmation_token=result.confirmation_token,
        expires_in_seconds=result.expires_in_seconds)
    response.status_code = _status(projected.reason)
    return projected


@router.post("/documents/{doc_name}/smart-cleanup", response_model=SmartCleanupOperationResponse,
             responses=_responses(SmartCleanupOperationResponse))
def execute(doc_name: DocumentName, body: SmartCleanupExecuteRequest, response: Response, service: Service):
    if service is None:
        raise HTTPException(503, "cleanup_service_unavailable")
    result = _project(service.execute(doc_name, body.operation_id,
        confirmation_token=body.confirmation_token, confirmed=True))
    response.status_code = _status(result.reason)
    return result


@router.get("/documents/{doc_name}/smart-cleanup/{operation_id}", response_model=SmartCleanupOperationResponse,
            responses=_responses(SmartCleanupOperationResponse))
def lookup(doc_name: DocumentName, operation_id: SmartCleanupOperationId, response: Response, service: Service):
    if service is None:
        raise HTTPException(503, "cleanup_service_unavailable")
    result = _project(service.lookup(doc_name, operation_id))
    response.status_code = _status(result.reason, lookup=True)
    return result
