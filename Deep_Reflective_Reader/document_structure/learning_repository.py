"""Storage-neutral learning port and internal source/batch contracts (LLRN-06).

All IDs/scopes are backend-resolved. No public transport DTOs or raw source
submission. A successful dispatch receipt is required before provider IO.
"""
from dataclasses import dataclass
from shared.learning_hint_assessment import HintAssessment
from typing import Callable, Protocol

from reading_preferences.models import WorkspaceScope, WorkspacePreferences, BookPreferences
from section_tasks.learning_cache import CacheKey, HintsKey, LearningResult
from section_tasks.learning_generation_policy import (
    BatchPlan, CacheEntry, CommandReceipt, GenerationBinding, GenerationCommand, GenerationOperation, ReadView, RequestDecision,
)
from shared.learning_results import (AlignedTranslation, GenerationProvenance, LearningHints,
    LearningResultContext, TranslatedSentence, VocabularyHint)
from shared.learning_source import (
    LearningSourceBinding, SentenceResource, SentenceResourceRef, integer, require, token, typed_tuple,
)

BindingValidator = Callable[[GenerationBinding, WorkspacePreferences, BookPreferences], None]


MAX_READ_ITEMS = 100  # Query bound, not a model/token/character limit.


class LearningStorageError(RuntimeError):
    pass


class LearningCommitUnknown(LearningStorageError):
    """Storage commit may have succeeded. Lookup/reconcile; do not resend provider work."""


@dataclass(frozen=True, slots=True)
class LearningBlock:
    source: LearningSourceBinding
    effective_text: str


@dataclass(frozen=True, slots=True)
class SourceCursor:
    document_id: str
    source_token: str
    offset: int


@dataclass(frozen=True, slots=True)
class SourcePage:
    blocks: tuple[LearningBlock, ...]
    next_cursor: SourceCursor | None


@dataclass(frozen=True, slots=True)
class SavedLearning:
    key: CacheKey
    view: ReadView
    sentences: SentenceResource | None


@dataclass(frozen=True, slots=True)
class SavedTranslation:
    value: SavedLearning
    operation: GenerationOperation


@dataclass(frozen=True, slots=True)
class TranslationPage:
    records: tuple[SavedTranslation, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class SentenceRead:
    state: str
    sentences: SentenceResource | None


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    operation: GenerationOperation
    entry_revision: int | None  # None when a newer operation owns the slot.


@dataclass(frozen=True, slots=True)
class BatchPayload:
    """Explicit partial payload; never a saved complete-result DTO."""
    component: str
    provenance: GenerationProvenance
    rows: tuple[TranslatedSentence, ...] = ()
    items: tuple[VocabularyHint, ...] = ()
    assessment: HintAssessment | None = None

    def __post_init__(self):
        require(self.component in ("translation", "hints"), "learning_component_invalid")
        require(type(self.provenance) is GenerationProvenance, "learning_provenance_invalid")
        typed_tuple(self.rows, TranslatedSentence)
        typed_tuple(self.items, VocabularyHint)
        require(self.assessment is None or (self.component == 'hints' and type(self.assessment) is HintAssessment
                and self.assessment.accepted_count == len(self.items)), 'learning_hint_assessment_invalid')
        require(not (self.items if self.component == "translation" else self.rows),
                "learning_batch_component_mismatch")

    def validate(self, operation: GenerationOperation, index: int,
                 sentences: SentenceResource, text: str) -> None:
        key = operation.binding.key
        integer(index)
        require(index < len(operation.plan.groups), "learning_batch_index_invalid")
        require(self.component == key.component and self.provenance.prompt_version == key.prompt_version,
                "learning_batch_component_mismatch")
        group = operation.plan.groups[index]
        if self.component == "translation":
            require(tuple(row.sentence_id for row in self.rows) == group, "learning_batch_alignment_invalid")
        else:
            require(all(item.sentence_id in group for item in self.items), "learning_batch_hint_invalid")
            context = LearningResultContext(key.resource, key.source_language, key.explanation_language,
                key.policy_version, operation.base_result_revision + 1, self.provenance)
            LearningHints(context, key.proficiency, self.items).validate_against(sentences, text)


@dataclass(frozen=True, slots=True)
class OperationWrite:
    operation_id: str
    expected_entry_revision: int
    fence: int
    now_ms: int
    action: str
    batch_index: int | None = None
    batch: BatchPayload | None = None
    result: LearningResult | None = None
    lease_until_ms: int | None = None
    reason: str | None = None
    outcome_known: bool | None = None

    def __post_init__(self):
        token(self.operation_id)
        integer(self.expected_entry_revision)
        integer(self.fence, minimum=1)
        integer(self.now_ms)
        allowed = {"dispatch": (), "checkpoint": ("batch_index", "batch"), "commit": ("result",),
            "renew": ("lease_until_ms",), "expire": (), "invalidate": (), "fail": ("reason", "outcome_known")}
        require(self.action in allowed, "learning_operation_action_invalid")
        for name in ("batch_index", "batch", "result", "lease_until_ms", "reason", "outcome_known"):
            require((getattr(self, name) is not None) == (name in allowed[self.action]),
                    "learning_operation_fields_invalid")
        if self.batch is not None:
            require(type(self.batch) is BatchPayload, "learning_batch_invalid")
            integer(self.batch_index)
        if self.result is not None:
            require(type(self.result) in (AlignedTranslation, LearningHints), "learning_result_invalid")
        if self.lease_until_ms is not None:
            integer(self.lease_until_ms)
        if self.reason is not None:
            token(self.reason)
            require(type(self.outcome_known) is bool, "learning_outcome_invalid")


class LearningRepository(Protocol):
    @property
    def scope(self) -> WorkspaceScope: ...

    def read_sources(self, document_id: str, *, limit: int = 50,
                     cursor: SourceCursor | None = None) -> SourcePage: ...

    def resolve_blocks(self, document_id: str, block_ids: tuple[str, ...]) -> tuple[LearningBlock, ...]: ...

    def inspect_operation(self, document_id: str, operation_id: str) -> OperationSnapshot | None: ...

    def read_sentences(self, document_id: str, references: tuple[SentenceResourceRef, ...]
                       ) -> tuple[SentenceRead, ...]: ...

    def read_batch(self, document_id: str, keys: tuple[CacheKey, ...]) -> tuple[SavedLearning, ...]: ...

    def read_translations(self, document_id: str, block_id: str, *, after: str = '',
                          limit: int = 50) -> TranslationPage: ...

    def save_sentences(self, document_id: str, sentences: SentenceResource) -> SentenceResource: ...

    def claim(self, document_id: str, command: GenerationCommand, *, sentences: SentenceResource,
              plan: BatchPlan | None, now_ms: int, lease_until_ms: int,
              validate_binding: BindingValidator | None = None) -> RequestDecision: ...

    def write_operation(self, document_id: str, command: OperationWrite, *,
                        validate_binding: BindingValidator | None = None) -> CacheEntry: ...

    def lookup_operation(self, document_id: str, operation_id: str) -> GenerationOperation | None: ...

    def lookup_command(self, document_id: str, command_id: str) -> CommandReceipt | None: ...

    def read_checkpoints(self, document_id: str, operation_id: str, *, offset: int = 0,
                         limit: int = 50) -> tuple[BatchPayload, ...]: ...
