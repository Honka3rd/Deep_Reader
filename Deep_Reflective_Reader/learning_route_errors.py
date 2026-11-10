"""Content-free errors for the learning/settings route family only."""
from dataclasses import asdict
import logging

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from api_learning_schemas import LearningErrorResponse
from learning import LearningModeUnavailable, LearningBusy
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError
from reading_preferences.effective import PreferenceValidationError
from reading_preferences.models import PreferenceCommitUnknown, PreferenceConflict, PreferenceError, PreferenceStorageError
from section_tasks.learning_translation_service import LearningTranslationError
from section_tasks.learning_hints_output import LearningHintsError
from shared.learning_source import LearningContractError

logger = logging.getLogger(__name__)

_CONFLICTS = {
    'learning_source_changed': 'learning_source_changed',
    'learning_binding_stale': 'learning_source_changed',
    'learning_settings_changed': 'learning_settings_changed',
    'learning_command_conflict': 'learning_command_conflict',
    'learning_operation_conflict': 'learning_operation_conflict',
    'learning_entry_revision_conflict': 'learning_operation_conflict',
    'learning_fence_conflict': 'learning_operation_conflict',
    'learning_recovery_plan_changed': 'learning_recovery_plan_changed',
    'preference_revision_conflict': 'preference_revision_conflict',
    'preference_snapshot_changed': 'preference_snapshot_changed',
}
_MISSING = {'learning_document_missing', 'learning_block_missing', 'learning_operation_missing'}


def error_detail(error, *, preference, mutation):
    invalid = 'preference_request_invalid' if preference else 'learning_request_invalid'
    unknown = {'kind': 'unknown', 'code': 'preference_commit_unknown' if preference else 'learning_commit_unknown',
               'recovery': 'read_preferences' if preference else 'lookup_command'}
    if isinstance(error, RequestValidationError):
        return 422, {'kind': 'invalid_request', 'code': invalid}
    if isinstance(error, (LearningCommitUnknown, PreferenceCommitUnknown)):
        return 503, unknown
    if isinstance(error, (LearningModeUnavailable, PreferenceValidationError)):
        return 422, {'kind': 'ineligible', 'code': 'preference_settings_invalid' if preference else 'learning_mode_unavailable',
                     'reasons': [asdict(reason) for reason in error.reasons]}
    if isinstance(error, PreferenceConflict):
        return 409, {'kind': 'conflict', 'code': _CONFLICTS.get(str(error), 'preference_revision_conflict')}
    if isinstance(error, FileNotFoundError):
        return 404, {'kind': 'not_found', 'code': 'learning_document_missing'}
    if isinstance(error, LearningBusy):
        return 503, {'kind': 'unavailable', 'code': 'learning_worker_busy'}
    if isinstance(error, PreferenceStorageError):
        return 503, {'kind': 'unavailable', 'code': 'preference_storage_unavailable'}
    if isinstance(error, LearningStorageError):
        return 503, {'kind': 'unavailable', 'code': 'preference_storage_unavailable' if preference else (
            'learning_storage_unavailable' if mutation else 'learning_read_failed')}
    if isinstance(error, (LearningTranslationError, LearningHintsError)):
        if error.code.endswith('sentence_capacity_exceeded'):
            return 422, {'kind': 'unavailable', 'code': 'learning_capacity_exceeded'}
        return 503, {'kind': 'unavailable', 'code': 'learning_provider_unavailable'}
    if isinstance(error, LearningContractError):
        code = str(error)
        if code in _CONFLICTS:
            return 409, {'kind': 'conflict', 'code': _CONFLICTS[code]}
        if code in _MISSING:
            return 404, {'kind': 'not_found', 'code': code}
        # Only enumerated client/domain preconditions may be exposed as invalid.
        if code in {'learning_retry_requires_new_command', 'learning_cursor_invalid', 'learning_read_limit_invalid'}:
            return 422, {'kind': 'invalid_request', 'code': invalid}
    if isinstance(error, PreferenceError):
        return 422, {'kind': 'invalid_request', 'code': invalid}
    # Unexpected failures during a mutation may follow a committed write. Never
    # claim rollback, replay a command or return raw exception/provider details.
    if mutation:
        return 503, unknown
    return 503, {'kind': 'unavailable', 'code': 'preference_storage_unavailable' if preference else 'learning_read_failed'}


class LearningRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        preference = 'reading-preferences' in self.path
        mutation = self.path.endswith(('/apply', '/generate', '/reconcile')) or 'PUT' in self.methods

        async def guarded(request):
            try:
                response = await handler(request)
            except Exception as error:
                status, detail = error_detail(error, preference=preference, mutation=mutation)
                logger.warning('learning_http_failed operation=%s code=%s status=%s', self.name, detail['code'], status)
                response = JSONResponse(LearningErrorResponse(detail=detail).model_dump(), status_code=status)
            response.headers['Cache-Control'] = 'no-store'
            return response
        return guarded
