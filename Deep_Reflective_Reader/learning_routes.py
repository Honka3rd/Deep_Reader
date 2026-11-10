"""Preference/learning transport; application and repository owners retain policy."""
from contextlib import asynccontextmanager
import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from starlette.concurrency import run_in_threadpool

import api_learning_schemas as api
import api_learning_projection as project
from document_structure.learning_repository import LearningStorageError
from learning_route_errors import LearningRoute


router = APIRouter(route_class=LearningRoute, tags=['language-learning'],
    responses={status: {'model': api.LearningErrorResponse} for status in (400, 404, 409, 422, 503)})


@asynccontextmanager
async def learning_lifespan(app):
    # Publish one runtime before concurrent requests can resolve DI resources.
    # Construction does no provider/tokenizer work, migration or recovery.
    container = getattr(app.state, 'learning_container', None)
    try:
        if container is not None:
            app.state.learning_runtime = await run_in_threadpool(container.learning_runtime)
        yield
    finally:
        if container is not None:
            await run_in_threadpool(container.shutdown_resources)
        if hasattr(app.state, 'learning_runtime'):
            del app.state.learning_runtime


def get_learning_runtime(request: Request):
    runtime = getattr(request.app.state, 'learning_runtime', None)
    if runtime is None:
        raise LearningStorageError('learning_service_unavailable')
    return runtime


Runtime = Annotated[object, Depends(get_learning_runtime)]


@router.get('/reading-preferences/options', response_model=api.ReadingOptionsResponse)
def options():
    return project.reading_options_response()


@router.get('/reading-preferences/workspace', response_model=api.ReadingWorkspaceResponse)
def workspace_read(runtime: Runtime):
    return project.workspace_response(runtime.preferences.read_workspace())


@router.put('/reading-preferences/workspace', response_model=api.ReadingWorkspaceResponse)
def workspace_save(payload: api.ReadingWorkspaceSaveRequest, runtime: Runtime):
    return project.workspace_response(runtime.preferences.save_workspace(payload.native_language,
        expected_revision=payload.expected_revision, mutation_id=payload.mutation_id))


@router.post('/documents/reading-preferences/read', response_model=api.ReadingBookResponse)
def book_read(payload: api.ReadingBookReadRequest, runtime: Runtime):
    return project.book_response(runtime.service.read_book_preferences(payload.doc_name))


@router.post('/documents/reading-preferences/apply', response_model=api.ReadingBookResponse)
def book_apply(payload: api.ReadingBookApplyRequest, runtime: Runtime):
    previous = runtime.service.read_book_preferences(payload.doc_name)
    settings = payload.to_settings(previous.book)
    return project.book_response(runtime.preferences.save_book(payload.doc_name, settings,
        expected_revision=payload.expected_revision, expected_workspace_revision=payload.expected_workspace_revision,
        mutation_id=payload.mutation_id))


@router.post('/documents/reading-interactions/learning/read', response_model=api.LearningReadResponse)
def learning_read(payload: api.LearningReadRequest, runtime: Runtime):
    return project.learning_read_response(runtime.service.read(payload.doc_name, tuple(payload.content_block_ids)))


def submission(doc_name, value, service):
    receipt = service.lookup_command(doc_name, value.command_id) if value.outcome == 'reuse' else None
    status = service.status(doc_name, value.operation_id) if value.outcome == 'succeeded' else None
    return project.submission_response(doc_name, value, receipt=receipt, status=status)


@router.post('/documents/reading-interactions/learning/generate', response_model=api.LearningSubmissionResponse)
def learning_generate(payload: api.LearningGenerateRequest, runtime: Runtime):
    service = runtime.service
    source = service.source_binding(payload.doc_name, payload.content_block_id)
    payload.verify_source(source)
    result = service.request(payload.doc_name, payload.content_block_id, payload.component,
        command_id=payload.command_id, expected_source=source,
        expected_workspace_revision=payload.expected_workspace_revision,
        expected_book_revision=payload.expected_book_revision, intent=payload.intent,
        retry_operation_id=payload.retry_operation.operation_id if payload.retry_operation else None,
        acknowledge_possible_duplicate_cost=payload.acknowledge_possible_duplicate_cost)
    return submission(payload.doc_name, result, service)


@router.post('/documents/reading-interactions/learning/operations/read', response_model=api.LearningOperationResponse)
def operation_read(payload: api.LearningOperationRequest, runtime: Runtime):
    handle = payload.handle
    return project.operation_response(handle, runtime.service.status(handle.doc_name, handle.operation_id))


@router.post('/documents/reading-interactions/learning/operations/wait', response_model=api.LearningOperationResponse)
async def operation_wait(payload: api.LearningOperationRequest, runtime: Runtime, request: Request):
    handle = payload.handle
    async def disconnected():
        while (await request.receive())['type'] != 'http.disconnect':
            pass
    observation = asyncio.create_task(runtime.service.wait_for_completion(handle.doc_name, handle.operation_id))
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((observation, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if observation in done:
            return project.operation_response(handle, observation.result())
        return Response(status_code=499)
    finally:
        for task in (observation, disconnect):
            if not task.done():
                task.cancel()
        await asyncio.gather(observation, disconnect, return_exceptions=True)


@router.post('/documents/reading-interactions/learning/operations/reconcile', response_model=api.LearningSubmissionResponse)
def operation_reconcile(payload: api.LearningOperationRequest, runtime: Runtime):
    handle = payload.handle
    result = runtime.service.reconcile(handle.doc_name, handle.operation_id)
    return submission(handle.doc_name, result, runtime.service)


@router.post('/documents/reading-interactions/learning/operations/partial', response_model=api.LearningPartialReadResponse)
def operation_partial(payload: api.LearningPartialReadRequest, runtime: Runtime):
    handle = payload.handle
    status, batches = runtime.service.read_partial_snapshot(handle.doc_name, handle.operation_id,
        offset=payload.offset, limit=payload.limit)
    return project.partial_response(handle, status, batches, offset=payload.offset)


@router.post('/documents/reading-interactions/learning/commands/read', response_model=api.LearningCommandLookupResponse)
def command_read(payload: api.LearningCommandLookupRequest, runtime: Runtime):
    return project.command_lookup_response(payload.doc_name, payload.command_id,
        runtime.service.lookup_command(payload.doc_name, payload.command_id))
