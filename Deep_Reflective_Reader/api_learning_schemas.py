"""LLRN-14 transport contracts. No routes, storage, provider or implicit mutation.

Only pure validation and conversion live here. Current source, profile, settings,
operation ownership and admission must still be verified by app/repository.
"""
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from shared.learning_hint_assessment import HintAssessment

from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from reading_preferences.models import (
    BookPreferences, BookSettings, LanguageLearningSettings, identifier, language,
)
from shared.learning_source import LearningSourceBinding, require, text_value, token


def _token(value):
    token(value)
    text_value(value)
    return value


def _document(value):
    _token(value)
    return identifier(value)


def _language(value):
    language(value)
    return value


Identity = Annotated[str, Field(min_length=1, max_length=255), AfterValidator(_token)]
DocumentName = Annotated[str, Field(min_length=1, max_length=255), AfterValidator(_document)]
CommandId = Annotated[str, Field(min_length=1, max_length=128), AfterValidator(_token)]
PreferenceMutationId = Annotated[CommandId, AfterValidator(identifier)]
RegisteredLanguage = Annotated[str, Field(min_length=1, max_length=32,
    json_schema_extra={'enum': [c for c in LanguageProfileRegistry.get_supported_language_codes() if c != 'unknown']}),
    AfterValidator(_language)]
Revision = Annotated[int, Field(ge=0)]
PositiveRevision = Annotated[int, Field(ge=1)]
SourceToken = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$', min_length=64, max_length=64)]
Text = Annotated[str, AfterValidator(lambda value: (text_value(value), value)[1])]
Component = Literal['translation', 'hints']
OperationState = Literal['claimed', 'running', 'partial', 'succeeded', 'failed', 'unknown', 'stale']
ViewState = Literal['missing', 'ready', 'pending', 'partial', 'failed', 'stale', 'unknown', 'unavailable']
PreferenceReason = Literal['profile_missing', 'profile_invalid', 'profile_unavailable',
    'source_language_required', 'source_language_invalid', 'native_language_required',
    'native_language_invalid', 'same_language', 'proficiency_required',
    'proficiency_language_mismatch', 'proficiency_invalid', 'settings_unsupported',
    'mode_unknown', 'book_deleted', 'learning_mode_not_active']


class LearningTransportDTO(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class ReadingBookReadRequest(LearningTransportDTO):
    doc_name: DocumentName


class ReadingWorkspaceSaveRequest(LearningTransportDTO):
    native_language: RegisteredLanguage | None
    expected_revision: Revision
    mutation_id: PreferenceMutationId


class NormalModeSelection(LearningTransportDTO):
    mode_id: Literal['normal']
    version: Literal['1']


class LearningModeSelection(LearningTransportDTO):
    mode_id: Literal['language_learning']
    version: Literal['1']
    language_code: RegisteredLanguage
    scheme_id: Identity
    scheme_version: Identity
    level_id: Identity

    @model_validator(mode='after')
    def valid_selection(self):
        LanguageProficiencyRegistry.resolve_selection(language_code=self.language_code,
            scheme_id=self.scheme_id, scheme_version=self.scheme_version, level_id=self.level_id)
        return self

    def to_domain(self):
        return LanguageLearningSettings(self.language_code, self.scheme_id, self.scheme_version, self.level_id)


ModeSelection = Annotated[NormalModeSelection | LearningModeSelection, Field(discriminator='mode_id')]


class ReadingBookApplyRequest(ReadingBookReadRequest):
    expected_revision: Revision
    expected_workspace_revision: Revision
    mutation_id: PreferenceMutationId
    selection: ModeSelection
    # Omission retains the previous override; explicit null selects saved profile.
    source_language_override: RegisteredLanguage | None = None

    @model_validator(mode='after')
    def matching_source(self):
        if isinstance(self.selection, LearningModeSelection) and self.source_language_override is not None:
            require(self.selection.language_code == self.source_language_override, 'preference_source_mismatch')
        return self

    def to_settings(self, previous: BookPreferences) -> BookSettings:
        """Merge only against server-read state; repository rechecks revisions/retention."""
        require(previous.document_id == self.doc_name, 'preference_document_mismatch')
        require(previous.revision == self.expected_revision, 'preference_revision_conflict')
        modes = previous.settings.modes
        if isinstance(self.selection, LearningModeSelection):
            modes = tuple(m for m in modes if m.mode_id != self.selection.mode_id) + (self.selection.to_domain(),)
        override = (self.source_language_override if 'source_language_override' in self.model_fields_set
                    else previous.settings.source_language_override)
        return BookSettings(self.selection.mode_id, override, modes)


class ReadingPreferenceIssue(LearningTransportDTO):
    code: PreferenceReason
    field: Identity
    message: Text


class ReadingModeOption(LearningTransportDTO):
    mode_id: Identity
    label: Text
    description: Text
    settings_version: Identity
    eligible: bool
    reasons: list[ReadingPreferenceIssue]

    @model_validator(mode='after')
    def coherent(self):
        require(self.eligible == (not self.reasons), 'preference_eligibility_mismatch')
        return self


class ReadingProficiencyLevel(LearningTransportDTO):
    level_id: Identity
    order: PositiveRevision
    label: Text
    reading_description: Text


class ReadingProficiencyScheme(LearningTransportDTO):
    language_code: RegisteredLanguage
    scheme_id: Identity
    version: Identity
    label: Text
    scope_note: Text
    levels: list[ReadingProficiencyLevel]


class ReadingOptionsResponse(LearningTransportDTO):
    language_codes: list[RegisteredLanguage]
    # All supported schemes let the settings UI change source without saving first.
    proficiency_schemes: list[ReadingProficiencyScheme]


class ReadingWorkspaceResponse(LearningTransportDTO):
    revision: Revision
    native_language: Identity | None


class ReadingSavedMode(LearningTransportDTO):
    """Unsupported stored data is a descriptor, never arbitrary client JSON."""
    mode_id: Identity
    version: Identity
    supported: bool
    selection: LearningModeSelection | None

    @model_validator(mode='after')
    def coherent(self):
        require(self.supported == (self.selection is not None), 'preference_saved_mode_mismatch')
        if self.selection:
            require((self.mode_id, self.version) == (self.selection.mode_id, self.selection.version),
                    'preference_saved_mode_mismatch')
        return self


class ReadingBookResponse(LearningTransportDTO):
    doc_name: DocumentName
    revision: Revision
    workspace: ReadingWorkspaceResponse
    active_mode: Identity
    effective_mode: Literal['normal', 'language_learning'] | None
    source_language_override: Identity | None
    source_language: RegisteredLanguage | None
    source_origin: Literal['profile', 'override']
    active_reasons: list[ReadingPreferenceIssue]
    modes: list[ReadingModeOption]
    saved_settings: list[ReadingSavedMode]

    @model_validator(mode='after')
    def coherent(self):
        require((self.effective_mode is None) == bool(self.active_reasons), 'preference_effective_mode_mismatch')
        if self.effective_mode is not None:
            require(self.effective_mode == self.active_mode, 'preference_effective_mode_mismatch')
        require(len({m.mode_id for m in self.modes}) == len(self.modes), 'preference_modes_duplicate')
        require(len({m.mode_id for m in self.saved_settings}) == len(self.saved_settings), 'preference_modes_duplicate')
        active = next((m for m in self.modes if m.mode_id == self.active_mode), None)
        if active is not None:
            require(active.reasons == self.active_reasons, 'preference_active_reasons_mismatch')
        if self.effective_mode == 'language_learning':
            require(self.source_language is not None and self.workspace.native_language is not None
                    and self.source_language != self.workspace.native_language, 'preference_languages_invalid')
            language(self.workspace.native_language)
            learning = next((m.selection for m in self.saved_settings if m.mode_id == 'language_learning'), None)
            require(learning is not None and learning.language_code == self.source_language,
                    'preference_learning_settings_required')
        return self


def learning_source_token(source: LearningSourceBinding) -> str:
    """Versioned opaque comparison token, NOT authorization or a client source plan."""
    body = json.dumps(['learning-api-source-v1', source.to_dict()], ensure_ascii=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(body).hexdigest()


class LearningOperationHandle(LearningTransportDTO):
    doc_name: DocumentName
    operation_id: CommandId


class LearningReadRequest(ReadingBookReadRequest):
    content_block_ids: list[Identity] = Field(min_length=1, max_length=50)

    @model_validator(mode='after')
    def unique_blocks(self):
        require(len(set(self.content_block_ids)) == len(self.content_block_ids), 'learning_blocks_duplicate')
        return self


class LearningGenerateRequest(ReadingBookReadRequest):
    content_block_id: Identity
    component: Component
    command_id: CommandId
    expected_source_token: SourceToken
    expected_workspace_revision: Revision
    expected_book_revision: Revision
    intent: Literal['get_or_generate', 'regenerate', 'retry'] = 'get_or_generate'
    retry_operation: LearningOperationHandle | None = None
    acknowledge_possible_duplicate_cost: bool = False

    @model_validator(mode='after')
    def retry_contract(self):
        if self.intent == 'retry':
            require(self.retry_operation is not None, 'learning_retry_target_required')
            require(self.retry_operation.doc_name == self.doc_name, 'learning_document_mismatch')
            require(self.retry_operation.operation_id != self.command_id, 'learning_retry_requires_new_command')
        else:
            require(self.retry_operation is None and not self.acknowledge_possible_duplicate_cost,
                    'learning_retry_fields_invalid')
        return self

    def verify_source(self, source: LearningSourceBinding) -> None:
        require((source.document_id, source.content_block_id) == (self.doc_name, self.content_block_id)
                and learning_source_token(source) == self.expected_source_token, 'learning_source_changed')


class LearningOperationRequest(LearningTransportDTO):
    handle: LearningOperationHandle


class LearningCommandLookupRequest(ReadingBookReadRequest):
    command_id: CommandId


class LearningPartialReadRequest(LearningOperationRequest):
    offset: Revision = 0
    limit: Annotated[int, Field(ge=1, le=100)] = 50


class LearningSpan(LearningTransportDTO):
    start: Revision
    end: PositiveRevision

    @model_validator(mode='after')
    def ordered(self):
        require(self.start < self.end, 'learning_span_invalid')
        return self


class LearningSentence(LearningSpan):
    sentence_id: Identity | None  # None is an exact whitespace gap, not a synthetic sentence.
    text: str

    @model_validator(mode='after')
    def projection(self):
        text_value(self.text, nonempty=False)
        require(len(self.text) == self.end-self.start, 'learning_span_text_mismatch')
        require(self.text.isspace() if self.sentence_id is None else bool(self.text.strip()),
                'learning_sentence_invalid')
        return self


class LearningTranslatedRow(LearningTransportDTO):
    sentence_id: Identity
    translated_text: Text


class LearningHint(LearningTransportDTO):
    sentence_id: Identity
    span: LearningSpan
    source_form: Text
    language_code: RegisteredLanguage
    kind: Literal['word', 'phrase', 'idiom']
    meaning: Text
    lemma: Text | None
    part_of_speech: Text | None


class LearningTranslationResult(LearningTransportDTO):
    component: Literal['translation']
    result_revision: PositiveRevision
    source_language: RegisteredLanguage
    explanation_language: RegisteredLanguage
    rows: list[LearningTranslatedRow]

    @model_validator(mode='after')
    def coherent(self):
        require(self.source_language != self.explanation_language, 'learning_languages_equal')
        require(len({r.sentence_id for r in self.rows}) == len(self.rows), 'learning_alignment_duplicate')
        return self


class LearningHintIssue(LearningTransportDTO):
    code: Identity
    count: PositiveRevision


class LearningHintAssessment(LearningTransportDTO):
    candidate_count: Revision
    accepted_count: Revision
    filtered_count: Revision
    rejected_count: Revision
    omitted_sentences: Revision
    issues: list[LearningHintIssue]

    @model_validator(mode='after')
    def coherent(self):
        HintAssessment.from_dict(self.model_dump())
        return self


class LearningHintsResult(LearningTransportDTO):
    component: Literal['hints']
    result_revision: PositiveRevision
    source_language: RegisteredLanguage
    explanation_language: RegisteredLanguage
    proficiency: LearningModeSelection
    items: list[LearningHint]
    assessment: LearningHintAssessment | None = None

    @model_validator(mode='after')
    def coherent(self):
        require(self.source_language != self.explanation_language, 'learning_languages_equal')
        require(self.proficiency.language_code == self.source_language, 'learning_proficiency_mismatch')
        require(all(i.language_code == self.source_language for i in self.items), 'learning_hint_language_mismatch')
        require(self.assessment is None or self.assessment.accepted_count == len(self.items), 'learning_hint_counts_invalid')
        identities = [(i.sentence_id, i.span.start, i.span.end, i.kind) for i in self.items]
        require(len(set(identities)) == len(identities), 'learning_hint_duplicate')
        return self


LearningSavedResult = Annotated[LearningTranslationResult | LearningHintsResult, Field(discriminator='component')]


class LearningComponentView(LearningTransportDTO):
    component: Component
    state: ViewState
    saved: LearningSavedResult | None
    operation: LearningOperationHandle | None
    operation_state: OperationState | None

    @model_validator(mode='after')
    def coherent(self):
        require((self.operation is None) == (self.operation_state is None), 'learning_operation_missing')
        require((self.state == 'ready') == (self.saved is not None), 'learning_ready_requires_saved')
        if self.saved:
            require(self.saved.component == self.component, 'learning_component_mismatch')
        else:
            expected = {'claimed': 'pending', 'running': 'pending', 'succeeded': 'unavailable'}
            if self.operation_state is not None:
                require(self.state == expected.get(self.operation_state, self.operation_state), 'learning_state_mismatch')
            else:
                require(self.state in ('missing', 'stale', 'unavailable'), 'learning_state_mismatch')
        return self


class LearningBlockResponse(LearningTransportDTO):
    content_block_id: Identity
    source_token: SourceToken
    source_length: Revision
    sentences: list[LearningSentence] | None
    components: list[LearningComponentView] = Field(max_length=2)

    @model_validator(mode='after')
    def aligned(self):
        if self.sentences is None:
            require(not self.components, 'learning_inactive_has_results')
            return self
        end = 0
        ids = []
        for sentence in self.sentences:
            require(sentence.start == end, 'learning_source_gap')
            end = sentence.end
            if sentence.sentence_id is not None:
                ids.append(sentence.sentence_id)
        require(end == self.source_length and len(set(ids)) == len(ids), 'learning_source_projection_invalid')
        require({c.component for c in self.components} == {'translation', 'hints'}, 'learning_components_invalid')
        source = ''.join(s.text for s in self.sentences)
        spans = {s.sentence_id: s for s in self.sentences if s.sentence_id is not None}
        for view in self.components:
            saved = view.saved
            if isinstance(saved, LearningTranslationResult):
                require([r.sentence_id for r in saved.rows] == ids, 'learning_alignment_invalid')
            elif isinstance(saved, LearningHintsResult):
                for hint in saved.items:
                    sentence = spans.get(hint.sentence_id)
                    require(sentence is not None and sentence.start <= hint.span.start < hint.span.end <= sentence.end
                            and source[hint.span.start:hint.span.end] == hint.source_form, 'learning_hint_source_mismatch')
        return self


class LearningReadResponse(LearningTransportDTO):
    doc_name: DocumentName
    settings: ReadingBookResponse
    blocks: list[LearningBlockResponse] = Field(max_length=50)

    @model_validator(mode='after')
    def binding(self):
        require(self.settings.doc_name == self.doc_name, 'learning_document_mismatch')
        require(len({b.content_block_id for b in self.blocks}) == len(self.blocks), 'learning_blocks_duplicate')
        for block in self.blocks:
            require((block.sentences is not None) == (self.settings.effective_mode == 'language_learning'),
                    'learning_effective_mode_mismatch')
            for view in block.components:
                if view.operation:
                    require(view.operation.doc_name == self.doc_name, 'learning_document_mismatch')
                if view.saved:
                    require((view.saved.source_language, view.saved.explanation_language)
                            == (self.settings.source_language, self.settings.workspace.native_language),
                            'learning_languages_mismatch')
                    if isinstance(view.saved, LearningHintsResult):
                        selection = next(m.selection for m in self.settings.saved_settings
                                         if m.mode_id == 'language_learning')
                        require(view.saved.proficiency == selection, 'learning_proficiency_mismatch')
        return self


class LearningSubmissionResponse(LearningTransportDTO):
    doc_name: DocumentName
    command_id: CommandId
    outcome: Literal['claimed', 'reuse', 'coalesced', 'reconcile', 'retry_required', 'no_content',
                     'denied', 'unavailable', 'unknown', 'missing', 'superseded', 'stale', 'failed',
                     'succeeded', 'resuming']
    operation: LearningOperationHandle | None
    replay: bool = False
    reason: Identity | None = None
    result_revision: PositiveRevision | None = None

    @model_validator(mode='after')
    def coherent(self):
        if self.operation:
            require(self.operation.doc_name == self.doc_name, 'learning_document_mismatch')
        if self.outcome in ('claimed', 'coalesced', 'reconcile', 'retry_required', 'resuming', 'succeeded', 'failed', 'stale', 'superseded'):
            require(self.operation is not None, 'learning_operation_required')
        if self.outcome in ('reuse', 'no_content', 'denied', 'missing'):
            require(self.operation is None, 'learning_operation_forbidden')
        require((self.reason is not None) == (self.outcome == 'denied'), 'learning_admission_reason_invalid')
        require((self.result_revision is not None) == (self.outcome in ('reuse', 'succeeded')),
                'learning_committed_revision_required')
        return self


class LearningOperationResponse(LearningTransportDTO):
    handle: LearningOperationHandle
    state: OperationState | Literal['missing']
    durable_state: OperationState | None
    component: Component | None
    content_block_id: Identity | None
    source_token: SourceToken | None
    source_current: bool
    settings_current: bool
    owns_current_slot: bool
    completed_batches: Revision
    total_batches: Revision
    result_revision: PositiveRevision | None
    outcome_uncertain: bool
    failure_code: Literal['provider_rejected', 'invalid_output', 'provider_outcome_unknown',
                          'worker_stopped', 'lease_expired', 'binding_changed'] | None

    @model_validator(mode='after')
    def coherent(self):
        require(self.completed_batches <= self.total_batches, 'learning_progress_invalid')
        fields = (self.durable_state, self.component, self.content_block_id, self.source_token)
        if self.state == 'missing':
            require(all(v is None for v in fields) and self.result_revision is None
                    and self.failure_code is None and not any((self.source_current, self.settings_current,
                    self.owns_current_slot, self.outcome_uncertain, self.total_batches)), 'learning_missing_operation_invalid')
        else:
            require(all(v is not None for v in fields) and self.total_batches > 0, 'learning_operation_incomplete')
            require(self.state == (self.durable_state if self.source_current and self.settings_current else 'stale'),
                    'learning_status_binding_mismatch')
            require((self.result_revision is not None) == (self.durable_state == 'succeeded'),
                    'learning_result_revision_invalid')
            if self.durable_state != 'stale':
                require(self.outcome_uncertain == (self.durable_state in ('running', 'unknown')),
                        'learning_outcome_state_invalid')
            if self.durable_state == 'succeeded':
                require(self.completed_batches == self.total_batches, 'learning_result_incomplete')
            elif self.durable_state == 'claimed':
                require(self.completed_batches == 0, 'learning_checkpoint_state_invalid')
            elif self.durable_state == 'partial':
                require(self.completed_batches > 0, 'learning_checkpoint_state_invalid')
            elif self.durable_state == 'running':
                require(self.completed_batches < self.total_batches, 'learning_batch_already_complete')
        return self


class LearningRequestError(LearningTransportDTO):
    kind: Literal['invalid_request']
    code: Literal['learning_request_invalid', 'preference_request_invalid']


class LearningConflictError(LearningTransportDTO):
    kind: Literal['conflict']
    code: Literal['learning_source_changed', 'learning_settings_changed', 'learning_command_conflict',
                  'learning_operation_conflict', 'learning_recovery_plan_changed', 'preference_revision_conflict',
                  'preference_snapshot_changed']


class LearningUnavailableError(LearningTransportDTO):
    kind: Literal['unavailable']
    code: Literal['learning_read_failed', 'learning_worker_busy', 'learning_capacity_exceeded',
                  'learning_provider_unavailable', 'learning_storage_unavailable', 'preference_storage_unavailable']


class LearningEligibilityError(LearningTransportDTO):
    kind: Literal['ineligible']
    code: Literal['learning_mode_unavailable', 'preference_settings_invalid']
    reasons: list[ReadingPreferenceIssue] = Field(min_length=1)


class LearningNotFoundError(LearningTransportDTO):
    kind: Literal['not_found']
    code: Literal['learning_document_missing', 'learning_block_missing', 'learning_operation_missing']


class LearningUnknownError(LearningTransportDTO):
    kind: Literal['unknown']
    code: Literal['learning_commit_unknown', 'preference_commit_unknown']
    # Caller must read/reconcile; transport success/failure cannot imply rollback.
    recovery: Literal['lookup_command', 'read_preferences']

    @model_validator(mode='after')
    def coherent(self):
        require(self.recovery == ('lookup_command' if self.code == 'learning_commit_unknown' else 'read_preferences'),
                'learning_recovery_mismatch')
        return self


LearningErrorDetail = Annotated[LearningRequestError | LearningConflictError | LearningUnavailableError |
    LearningEligibilityError | LearningNotFoundError | LearningUnknownError, Field(discriminator='kind')]


class LearningErrorResponse(LearningTransportDTO):
    detail: LearningErrorDetail


class LearningCommandLookupResponse(LearningTransportDTO):
    doc_name: DocumentName
    command_id: CommandId
    found: bool
    receipt: LearningSubmissionResponse | None

    @model_validator(mode='after')
    def coherent(self):
        require(self.found == (self.receipt is not None), 'learning_receipt_missing')
        if self.receipt:
            require((self.receipt.doc_name, self.receipt.command_id) == (self.doc_name, self.command_id),
                    'learning_command_mismatch')
        return self


class LearningTranslationBatch(LearningTransportDTO):
    component: Literal['translation']
    batch_index: Revision
    rows: list[LearningTranslatedRow]


class LearningHintsBatch(LearningTransportDTO):
    component: Literal['hints']
    batch_index: Revision
    items: list[LearningHint]
    assessment: LearningHintAssessment | None = None

    @model_validator(mode='after')
    def coherent(self):
        require(self.assessment is None or self.assessment.accepted_count == len(self.items), 'learning_hint_counts_invalid')
        return self


LearningPartialBatch = Annotated[LearningTranslationBatch | LearningHintsBatch, Field(discriminator='component')]


class LearningPartialReadResponse(LearningTransportDTO):
    handle: LearningOperationHandle
    component: Component
    source_token: SourceToken
    offset: Revision
    completed_batches: Revision
    total_batches: PositiveRevision
    batches: list[LearningPartialBatch] = Field(max_length=100)
    next_offset: Revision | None

    @model_validator(mode='after')
    def coherent(self):
        require(self.completed_batches <= self.total_batches, 'learning_progress_invalid')
        require(all(b.component == self.component and b.batch_index == self.offset+i
                    for i, b in enumerate(self.batches)), 'learning_partial_order_invalid')
        require(not self.batches or self.offset+len(self.batches) <= self.completed_batches,
                'learning_partial_range_invalid')
        end = self.offset+len(self.batches)
        require(self.next_offset == (end if end < self.completed_batches else None), 'learning_partial_cursor_invalid')
        require(bool(self.batches) or self.offset >= self.completed_batches, 'learning_checkpoint_missing')
        return self
