"""Current-page learning projection and transaction-bound effective-setting guards."""
from dataclasses import dataclass

from document_structure.learning_repository import LearningBlock, SavedLearning, OperationSnapshot, MAX_READ_ITEMS
from reading_preferences.effective import EffectiveBookPreferences, issue
from reading_preferences.models import PreferenceError, identifier
from section_tasks.learning_cache import HintsKey, TranslationKey
from section_tasks.learning_generation_policy import GenerationBinding
from section_tasks.learning_sentence_segmenter import segment_learning_block
from shared.learning_results import LearningProficiencyRef
from shared.learning_source import LearningContractError, SentenceResource, require


@dataclass(frozen=True, slots=True)
class LearningBlockView:
    block: LearningBlock
    sentences: SentenceResource | None
    components: tuple[SavedLearning, ...]


@dataclass(frozen=True, slots=True)
class LearningPage:
    settings: EffectiveBookPreferences
    blocks: tuple[LearningBlockView, ...]


@dataclass(frozen=True, slots=True)
class LearningSubmission:
    outcome: str
    command_id: str
    operation_id: str | None
    replay: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LearningStatus:
    snapshot: OperationSnapshot | None
    state: str
    source_current: bool
    settings_current: bool


class LearningModeUnavailable(LearningContractError):
    def __init__(self, reasons):
        super().__init__('learning_mode_unavailable')
        self.reasons = reasons


class LearningContext:
    def __init__(self, repository, preferences, translation, hints):
        require(repository.scope == preferences.scope, 'learning_scope_mismatch')
        self.repository, self.preferences = repository, preferences
        self.services = {'translation': translation, 'hints': hints}

    @staticmethod
    def eligible(settings):
        if settings.effective_mode != 'language_learning':
            raise LearningModeUnavailable(settings.active.reasons or (issue('learning_mode_not_active'),))

    @staticmethod
    def proficiency(settings):
        value = settings.proficiency
        require(value is not None, 'learning_proficiency_invalid')
        return LearningProficiencyRef(value.scheme.scheme_id, value.scheme.version, value.level.level_id)

    def validate_binding(self, binding, workspace, book):
        """Called inside repository source/preference locks; no nested storage writes."""
        if binding.active_mode == 'translation':
            require(type(binding.key) is TranslationKey and not book.deleted
                    and binding.workspace_revision == binding.book_revision == 0,
                    'learning_binding_stale')
            binding.key.validate_selection()
            return
        try:
            settings = self.preferences.resolve_snapshot(workspace, book)
            self.eligible(settings)
            key = binding.key
            require((workspace.revision, book.revision, book.settings.active_mode)
                    == (binding.workspace_revision, binding.book_revision, binding.active_mode),
                    'learning_settings_changed')
            require((key.source_language, key.explanation_language)
                    == (settings.source.code, workspace.native_language), 'learning_settings_changed')
            if type(key) is HintsKey:
                require(key.proficiency == self.proficiency(settings), 'learning_settings_changed')
        except PreferenceError:
            raise LearningContractError('learning_settings_changed') from None

    def arguments(self, settings, sentences):
        return dict(sentences=sentences, source_language=settings.source.code,
                    explanation_language=settings.workspace.native_language)

    def key(self, component, settings, sentences):
        require(component in self.services, 'learning_component_invalid')
        args = self.arguments(settings, sentences)
        if component == 'hints':
            args['proficiency'] = self.proficiency(settings)
        return self.services[component].key_for(**args)

    def prepare(self, component, settings, block, sentences):
        args = self.arguments(settings, sentences)
        if component == 'hints':
            args['proficiency'] = self.proficiency(settings)
        return self.services[component].prepare(effective_text=block.effective_text, **args)

    def selection(self, document_id, block_id):
        identifier(document_id)
        settings = self.preferences.read_book(document_id)
        self.eligible(settings)
        block, = self.repository.resolve_blocks(document_id, (block_id,))
        sentences = segment_learning_block(source=block.source, effective_text=block.effective_text,
                                           source_language=settings.source.code)
        return settings, block, sentences

    def read(self, document_id, block_ids):
        identifier(document_id)
        # Two components per block fit in a single bounded repository read.
        require(type(block_ids) is tuple and len(block_ids) <= MAX_READ_ITEMS // 2,
                'learning_page_limit_invalid')
        settings = self.preferences.read_book(document_id)
        blocks = self.repository.resolve_blocks(document_id, block_ids)
        if settings.effective_mode != 'language_learning':
            return LearningPage(settings, tuple(LearningBlockView(b, None, ()) for b in blocks))
        resources = tuple(segment_learning_block(source=b.source, effective_text=b.effective_text,
            source_language=settings.source.code) for b in blocks)
        keys = tuple(self.key(component, settings, resource) for resource in resources
                     for component in ('translation', 'hints'))
        saved = self.repository.read_batch(document_id, keys) if keys else ()
        # A stale source observed between snapshots is a conflict, never a miss.
        require(all(row.view.state != 'stale' or row.view.operation_state == 'stale' for row in saved),
                'learning_source_changed')
        require(self.preferences.read_book(document_id) == settings, 'learning_settings_changed')
        return LearningPage(settings, tuple(LearningBlockView(block, resource, saved[index*2:index*2+2])
            for index, (block, resource) in enumerate(zip(blocks, resources))))

    def status(self, document_id, operation_id):
        identifier(document_id)
        snapshot = self.repository.inspect_operation(document_id, operation_id)
        if snapshot is None:
            return LearningStatus(None, 'missing', False, False)
        op = snapshot.operation
        try:
            block, = self.repository.resolve_blocks(document_id, (op.binding.key.resource.source.content_block_id,))
            source_current = block.source == op.binding.key.resource.source
        except FileNotFoundError:
            source_current = False
        except LearningContractError as error:
            if str(error) != 'learning_block_missing':
                raise
            source_current = False
        if op.binding.active_mode == 'translation':
            return LearningStatus(snapshot, op.state if source_current else 'stale', source_current, True)
        settings = self.preferences.read_book(document_id)
        try:
            # The service already resolved this read snapshot; avoid extra profile IO.
            self.eligible(settings)
            require(GenerationBinding(op.binding.key, settings.workspace.revision, settings.book.revision,
                settings.book.settings.active_mode) == op.binding, 'learning_settings_changed')
            require((settings.source.code, settings.workspace.native_language)
                    == (op.binding.key.source_language, op.binding.key.explanation_language), 'learning_settings_changed')
            if type(op.binding.key) is HintsKey:
                require(op.binding.key.proficiency == self.proficiency(settings), 'learning_settings_changed')
            settings_current = True
        except LearningContractError:
            settings_current = False
        current = source_current and settings_current
        return LearningStatus(snapshot, op.state if current else 'stale', source_current, settings_current)
