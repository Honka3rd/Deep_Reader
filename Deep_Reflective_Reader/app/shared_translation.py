"""Ordinary translation adapter over the canonical aligned translation runtime.

No paragraph output is written to document artifacts. The accepted worker owns
execution; this synchronous compatibility endpoint only waits for notification.
"""
from uuid import uuid4

from document_structure.learning_repository import LearningCommitUnknown, OperationSnapshot
from section_tasks.aligned_translation_projection import paragraph_record
from section_tasks.learning_generation_policy import GenerationBinding, GenerationCommand
from section_tasks.learning_sentence_segmenter import segment_learning_block
from section_tasks.learning_translation_prompt import PROMPT_VERSION
from section_tasks.translation_contracts import TranslationError, TranslationResult, ensure_distinct_translation_languages
from section_tasks.translation_records import interactive_language, translation_snapshot
from shared.learning_source import LearningContractError, require


class SharedTranslation:
    def __init__(self, documents, default_language, service, *, wait_seconds=120):
        self.documents, self.default_language, self.service = documents, default_language, service
        self.repository, self.worker = service.repository, service.worker
        self.wait_seconds = wait_seconds
        self.translator = service.context.services['translation']

    def catalog(self, doc, snapshot):
        cursor, records = '', []
        while True:
            page = self.repository.read_translations(doc, snapshot.target.content_block_id, after=cursor)
            for saved in page.records:
                value = saved.value
                sentences = segment_learning_block(source=value.key.resource.source,
                    effective_text=snapshot.target.content, source_language=value.key.source_language)
                current = self.translator.key_for(sentences=sentences,
                    source_language=value.key.source_language, explanation_language=value.key.explanation_language)
                if current == value.key:
                    require(value.key.resource.source.effective_content_hash == snapshot.content_hash,
                            'learning_source_changed')
                    records.append((value.key, paragraph_record(saved, snapshot.target.content)))
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        self.current(doc, snapshot)
        return records

    def current(self, doc, snapshot):
        current = translation_snapshot(self.documents.load_document(doc), snapshot.target.content_block_id)
        if current.binding != snapshot.binding or current.target.reading_revision != snapshot.target.reading_revision:
            raise TranslationError('translation_source_changed')

    def read_records(self, doc, snapshot, default_source):
        records = {r['target_language']: r for r in snapshot.records()}
        # Current-version aligned output takes precedence over legacy paragraph data.
        # The legacy UI exposes one record per target; prefer the current source default.
        aligned = sorted(self.catalog(doc, snapshot), key=lambda pair: (
            pair[1]['source_language'] == default_source, pair[1]['updated_at'], pair[0].cache_id))
        for _, record in aligned:
            records[record['target_language']] = record
        return [records[code] for code in sorted(records)]

    def generate(self, *, doc_name, content_block_id, target_language, source_language=None, mutation_id=None):
        target_language = interactive_language(target_language, field='target_language')
        source_language = (self.default_language(doc_name) if source_language is None else
                           interactive_language(source_language, field='source_language'))
        if source_language is None:
            raise TranslationError('translation_source_language_required')
        ensure_distinct_translation_languages(source_language, target_language)
        mutation_id = uuid4().hex if mutation_id is None else mutation_id
        if not isinstance(mutation_id, str) or not 1 <= len(mutation_id.strip()) <= 100:
            raise TranslationError('translation_request_invalid')
        snapshot = translation_snapshot(self.documents.load_document(doc_name), content_block_id)
        block, = self.repository.resolve_blocks(doc_name, (content_block_id,))
        require(block.effective_text == snapshot.target.content, 'learning_source_changed')
        sentences = segment_learning_block(source=block.source, effective_text=block.effective_text,
                                           source_language=source_language)
        key = self.translator.key_for(sentences=sentences, source_language=source_language,
                                      explanation_language=target_language)
        command_id = 'translation:' + mutation_id
        prior = self.repository.lookup_command(doc_name, command_id)
        if prior is not None and prior.command.binding.key != key:
            raise TranslationError('translation_mutation_conflict')
        for saved_key, record in self.catalog(doc_name, snapshot):
            if saved_key == key:
                return self.result(snapshot, record)
        # Historical paragraph-only output remains readable; it is not fabricated alignment.
        legacy = next((r for r in snapshot.records() if r['target_language'] == target_language
                       and r['source_language'] == source_language), None)
        if legacy is not None and prior is None:
            return self.result(snapshot, legacy, legacy=True)
        saved, = self.repository.read_batch(doc_name, (key,))
        require(saved.view.state not in ('unavailable', 'unsupported', 'read_error'), 'learning_read_not_verified')
        if prior is not None:
            operation_id = prior.operation_id
            if operation_id is None:
                raise TranslationError('translation_service_unavailable')
            # Repeating the explicit command may resume a safely claimed/partial
            # operation; reconcile never redispatches running or unknown work.
            self.service.reconcile(doc_name, operation_id)
        else:
            operation_id = self.claim(doc_name, block, sentences, key, command_id, saved)
        if operation_id is not None:
            self.observe(doc_name, operation_id)
        for saved_key, record in self.catalog(doc_name, snapshot):
            if saved_key == key:
                return self.result(snapshot, record)
        if operation_id is None:
            raise TranslationError('translation_service_unavailable')
        status = self.service.status(doc_name, operation_id)
        if status.state == 'stale':
            raise TranslationError('translation_source_changed')
        if status.state == 'failed':
            code = 'translation_invalid_output' if status.snapshot.operation.failure_code == 'invalid_output' else 'translation_provider_failed'
            raise TranslationError(code)
        # Includes timeout/unknown: do not fabricate a rejection or dispatch again.
        raise TranslationError('translation_outcome_unconfirmed')

    def claim(self, doc, block, sentences, key, command_id, saved):
        op = self.repository.inspect_operation(doc, saved.view.operation_id) if saved.view.operation_id else None
        retry = op is not None and op.operation.state in ('failed', 'stale') and not op.operation.outcome_uncertain
        command = GenerationCommand(command_id, GenerationBinding(key, 0, 0, 'translation'),
            'retry' if retry else 'get_or_generate', op.operation.operation_id if retry else None)
        reserved, handed_off = False, False
        try:
            prepared = None
            # Active/unknown work is inspected without provider preflight or a new slot.
            if saved.view.operation_state not in ('claimed', 'running', 'partial', 'unknown'):
                self.worker.reserve(); reserved = True
                prepared = self.translator.prepare(sentences=sentences, effective_text=block.effective_text,
                    source_language=key.source_language, explanation_language=key.explanation_language)
            now = self.service.clock()
            decision = self.repository.claim(doc, command, sentences=sentences,
                plan=prepared.plan if prepared else None, now_ms=now, lease_until_ms=now+self.worker.lease_ms,
                validate_binding=self.service.context.validate_binding)
            if decision.next_entry is not None:
                entry = decision.next_entry
                snapshot = OperationSnapshot(entry.operation, entry.revision)
                try:
                    self.worker.start(doc, prepared, snapshot); handed_off = True
                except Exception:
                    self.service.record_start_failure(doc, snapshot)
                    raise
            if decision.receipt.operation_id is None and decision.outcome != 'reuse':
                raise TranslationError('translation_service_unavailable')
            return decision.receipt.operation_id
        except LearningCommitUnknown:
            raise TranslationError('translation_outcome_unconfirmed') from None
        finally:
            if reserved and not handed_off:
                self.worker.release()

    def observe(self, doc, operation_id):
        completion = self.worker.completion
        future = completion.subscribe(doc, operation_id)
        try:
            status = self.service.status(doc, operation_id)
            if status.state in ('claimed', 'running', 'partial'):
                try:
                    future.result(timeout=self.wait_seconds)
                except TimeoutError:
                    pass
        finally:
            completion.unsubscribe(doc, operation_id, future)

    @staticmethod
    def result(snapshot, record, *, legacy=False):
        from app.translation_action import TranslationActionResult
        from section_tasks.translation_prompt import TRANSLATION_PROMPT_VERSION
        result = TranslationResult(record['source_language'], record['target_language'],
                                   record['status'], record['translated_text'])
        return TranslationActionResult(snapshot.target, result, snapshot.content_hash, record,
            TRANSLATION_PROMPT_VERSION if legacy else PROMPT_VERSION)
