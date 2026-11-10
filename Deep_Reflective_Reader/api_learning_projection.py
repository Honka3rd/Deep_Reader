"""Pure transport projections; callers supply already-authoritative domain reads.

Never performs IO, looks up a profile, revalidates admission or starts a worker.
"""
from dataclasses import asdict

from pydantic import ValidationError

import api_learning_schemas as api
from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.models import LanguageLearningSettings
from shared.learning_results import AlignedTranslation
from shared.learning_source import require


def reading_options_response():
    codes = [c for c in LanguageProfileRegistry.get_supported_language_codes() if c != 'unknown']
    schemes = []
    for code in codes:
        scheme = LanguageProficiencyRegistry.get_scheme(code)
        schemes.append(api.ReadingProficiencyScheme(language_code=code, scheme_id=scheme.scheme_id,
            version=scheme.version, label=scheme.label, scope_note=scheme.scope_note,
            levels=[api.ReadingProficiencyLevel(**asdict(level)) for level in scheme.levels]))
    return api.ReadingOptionsResponse(language_codes=codes, proficiency_schemes=schemes)


def workspace_response(value):
    return api.ReadingWorkspaceResponse(revision=value.revision, native_language=value.native_language)


def _issue(value):
    return api.ReadingPreferenceIssue(**asdict(value))


def book_response(value):
    eligibility = {m.mode_id: m for m in value.modes}
    saved = []
    for entry in value.book.settings.modes:
        selection = None
        if type(entry) is LanguageLearningSettings:
            try:
                selection = api.LearningModeSelection(mode_id=entry.mode_id, **asdict(entry))
            except ValidationError:
                # Stale stored registry/version is readable, not silently repaired.
                pass
        saved.append(api.ReadingSavedMode(mode_id=entry.mode_id, version=entry.version,
            supported=selection is not None, selection=selection))
    return api.ReadingBookResponse(doc_name=value.book.document_id, revision=value.book.revision,
        workspace=workspace_response(value.workspace), active_mode=value.book.settings.active_mode,
        effective_mode=value.effective_mode, source_language_override=value.book.settings.source_language_override,
        source_language=value.source.code, source_origin=value.source.origin,
        active_reasons=[_issue(r) for r in value.active.reasons], saved_settings=saved,
        modes=[api.ReadingModeOption(**asdict(m), eligible=eligibility[m.mode_id].eligible,
            reasons=[_issue(r) for r in eligibility[m.mode_id].reasons]) for m in value.catalog])


def _handle(doc_name, operation_id):
    return api.LearningOperationHandle(doc_name=doc_name, operation_id=operation_id) if operation_id else None


def _saved(result):
    if result is None:
        return None
    common = dict(result_revision=result.context.result_revision, source_language=result.context.source_language,
                  explanation_language=result.context.explanation_language)
    if type(result) is AlignedTranslation:
        return api.LearningTranslationResult(component='translation', **common,
            rows=[api.LearningTranslatedRow(**asdict(row)) for row in result.rows])
    return api.LearningHintsResult(component='hints', **common,
        proficiency=api.LearningModeSelection(mode_id='language_learning', version='1',
            language_code=result.context.source_language, **asdict(result.proficiency)),
        items=[api.LearningHint.model_validate(asdict(item)) for item in result.items],
        assessment=_assessment(result.assessment))


def _assessment(value):
    return (api.LearningHintAssessment.model_validate({**asdict(value),
        'issues': [asdict(issue) for issue in value.issues]}) if value is not None else None)


def learning_read_response(page):
    doc_name = page.settings.book.document_id
    blocks = []
    for view in page.blocks:
        source = view.block.source
        require(source.document_id == doc_name, 'learning_document_mismatch')
        sentences = ([api.LearningSentence(sentence_id=p.sentence_id, start=p.span.start,
            end=p.span.end, text=p.text) for p in view.sentences.project(view.block.effective_text)]
            if view.sentences is not None else None)
        components = []
        for saved in view.components:
            require(saved.key.resource.source == source, 'learning_source_changed')
            current = saved.view
            components.append(api.LearningComponentView(component=saved.key.component,
                state='pending' if current.state in ('claimed', 'running') else current.state,
                saved=_saved(current.saved), operation=_handle(doc_name, current.operation_id),
                operation_state=current.operation_state))
        blocks.append(api.LearningBlockResponse(content_block_id=source.content_block_id,
            source_token=api.learning_source_token(source), source_length=len(view.block.effective_text),
            sentences=sentences, components=components))
    return api.LearningReadResponse(doc_name=doc_name, settings=book_response(page.settings), blocks=blocks)


def operation_response(handle, status):
    if status.snapshot is None:
        return api.LearningOperationResponse(handle=handle, state='missing', durable_state=None, component=None,
            content_block_id=None, source_token=None, source_current=False, settings_current=False,
            owns_current_slot=False, completed_batches=0, total_batches=0, result_revision=None,
            outcome_uncertain=False, failure_code=None)
    op = status.snapshot.operation
    source = op.binding.key.resource.source
    require((handle.doc_name, handle.operation_id) == (source.document_id, op.operation_id), 'learning_document_mismatch')
    return api.LearningOperationResponse(handle=handle, state=status.state, durable_state=op.state,
        component=op.binding.key.component, content_block_id=source.content_block_id,
        source_token=api.learning_source_token(source), source_current=status.source_current,
        settings_current=status.settings_current, owns_current_slot=status.snapshot.entry_revision is not None,
        completed_batches=len(op.checkpoints), total_batches=len(op.plan.groups), result_revision=op.result_revision,
        outcome_uncertain=op.outcome_uncertain, failure_code=op.failure_code)


def submission_response(doc_name, submission, *, receipt=None, status=None):
    revision = None
    if receipt is not None:
        require(receipt.command.binding.key.resource.source.document_id == doc_name
                and receipt.command.command_id == submission.command_id, 'learning_command_mismatch')
        require((receipt.outcome, receipt.operation_id) == (submission.outcome, submission.operation_id),
                'learning_receipt_mismatch')
        revision = receipt.result_revision
    if submission.outcome == 'succeeded':
        require(status is not None and status.snapshot is not None, 'learning_committed_revision_required')
        op = status.snapshot.operation
        require(op.state == 'succeeded' and op.operation_id == submission.operation_id
                and op.binding.key.resource.source.document_id == doc_name, 'learning_operation_mismatch')
        revision = op.result_revision
    return api.LearningSubmissionResponse(doc_name=doc_name, command_id=submission.command_id,
        outcome=submission.outcome, operation=_handle(doc_name, submission.operation_id),
        replay=submission.replay, reason=submission.reason, result_revision=revision)


def command_lookup_response(doc_name, command_id, receipt):
    result = None
    if receipt is not None:
        from learning import LearningSubmission
        result = submission_response(doc_name, LearningSubmission(receipt.outcome, command_id,
            receipt.operation_id, True, receipt.admission_reason), receipt=receipt)
    return api.LearningCommandLookupResponse(doc_name=doc_name, command_id=command_id,
        found=receipt is not None, receipt=result)


def partial_response(handle, status, batches, *, offset=0):
    require(status.snapshot is not None and status.source_current and status.settings_current, 'learning_binding_stale')
    # Also checks document/operation identity before exposing any checkpoint.
    operation_response(handle, status)
    op = status.snapshot.operation
    component = op.binding.key.component
    projected = []
    for index, batch in enumerate(batches, offset):
        if component == 'translation':
            projected.append(api.LearningTranslationBatch(component=component, batch_index=index,
                rows=[api.LearningTranslatedRow(**asdict(row)) for row in batch.rows]))
        else:
            projected.append(api.LearningHintsBatch(component=component, batch_index=index,
                items=[api.LearningHint.model_validate(asdict(item)) for item in batch.items],
                assessment=_assessment(batch.assessment)))
    end = offset+len(projected)
    return api.LearningPartialReadResponse(handle=handle, component=component,
        source_token=api.learning_source_token(op.binding.key.resource.source), offset=offset,
        completed_batches=len(op.checkpoints), total_batches=len(op.plan.groups), batches=projected,
        next_offset=end if end < len(op.checkpoints) else None)
