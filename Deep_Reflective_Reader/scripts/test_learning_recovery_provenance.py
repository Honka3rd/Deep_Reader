"""LEARN-01: real file/SQL recovery, deterministic providers and real batch plans."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from scripts.test_learning_orchestration import (
    OrchestrationCases, FileLearningOrchestrationTests, SqlLearningOrchestrationTests,
)
from scripts.test_learning_translation_service import Provider
from scripts.test_learning_hints_service import Provider as HintsProvider
from section_tasks.learning_translation_service import LearningTranslationService, _prompt
from section_tasks.learning_hints_service import LearningHintsService
from section_tasks.learning_hints_prompt import build_prompt
from document_structure.learning_repository import LearningStorageError
from shared.learning_source import LearningContractError


class RecoveryCases:
    setup_service = OrchestrationCases.setup_service
    make_service = OrchestrationCases.make_service
    request = OrchestrationCases.request
    finish = OrchestrationCases.finish
    read = OrchestrationCases.read
    fault_write = OrchestrationCases.fault_write
    settings = OrchestrationCases.settings
    change = OrchestrationCases.change

    def prepare_partial(self, component, *, previous=False):
        self.service = self.make_service()
        self.op = component + '-partial'
        provider = self.trans_provider if component == 'translation' else self.hints_provider
        if previous:
            self.request(component, component + '-saved'); self.finish()
            self.service = self.make_service()
        settings, block, sentences = self.service.context.selection('book', self.block.source.content_block_id)
        prepared = self.service.context.prepare(component, settings, block, sentences)
        prompt = _prompt if component == 'translation' else build_prompt
        ceiling = max(len(prompt(prepared.key, sentences, block.effective_text, (s.sentence_id,)))
                      for s in sentences.sentences) + 256
        provider.caps = replace(provider.caps, max_input_tokens=ceiling)
        self.prepared = self.service.context.prepare(component, settings, block, sentences)
        self.assertEqual(len(self.prepared.plan.groups), 2)
        with self.fault_write('checkpoint'):
            self.request(component, self.op, intent='regenerate' if previous else 'get_or_generate')
            self.finish()
        snapshot = self.service.status('book', self.op).snapshot
        self.assertEqual(snapshot.operation.state, 'partial')
        self.assertEqual(len(snapshot.operation.checkpoints), 1)
        return snapshot

    def restart(self, component, *, model=None, provider_name='test'):
        provider = Provider() if component == 'translation' else HintsProvider()
        provider.caps = replace(self.prepared.capabilities, model_name=model or self.prepared.capabilities.model_name)
        if component == 'translation':
            self.trans_provider = provider
            self.translation = LearningTranslationService(provider, provider_name=provider_name)
        else:
            self.hints_provider = provider
            self.hints = LearningHintsService(provider, provider_name=provider_name)
        self.service = self.make_service()
        return provider

    def saved(self, component):
        return self.read().blocks[0].components[0 if component == 'translation' else 1].view.saved

    def test_changed_model_stops_before_dispatch_and_explicit_retry_succeeds(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                old = self.prepare_partial(component)
                parts = self.repo.read_checkpoints('book', self.op)
                receipt = self.repo.lookup_command('book', self.op)
                provider = self.restart(component, model='model-B')
                self.service.reconcile('book', self.op); self.finish()
                self.assertEqual(len(provider.calls), 0)
                failed = self.service.status('book', self.op).snapshot
                self.assertEqual((failed.operation.state, failed.operation.failure_code), ('failed', 'recovery_incompatible'))
                self.assertEqual(self.repo.read_checkpoints('book', self.op), parts)
                self.assertEqual(self.repo.lookup_command('book', self.op), receipt)
                self.now += 10001; self.service = self.make_service()
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'failed')
                self.assertEqual(self.request(component, self.op+'-read').outcome, 'retry_required')
                self.assertEqual(provider.calls, [])
                retry = self.request(component, self.op+'-retry', intent='retry', retry_operation_id=self.op)
                self.assertEqual(retry.outcome, 'claimed'); self.finish()
                self.assertEqual(len(provider.calls), 2)
                result = self.saved(component)
                self.assertEqual(result.context.provenance.model, 'model-B')
                self.assertGreater(self.service.status('book', retry.operation_id).snapshot.operation.fence, old.operation.fence)
                self.assertEqual(self.repo.read_checkpoints('book', self.op), parts)

    def test_changed_provider_preserves_previous_saved_and_reuses_across_models(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.prepare_partial(component, previous=True)
                saved = self.saved(component)
                provider = self.restart(component, provider_name='other-provider')
                self.service.reconcile('book', self.op); self.finish()
                self.assertEqual(provider.calls, [])
                self.assertEqual(self.service.status('book', self.op).state, 'failed')
                self.assertEqual(self.saved(component), saved)
                self.service = self.make_service()
                with patch.object(provider, 'get_model_capabilities', side_effect=AssertionError('reuse capacity')):
                    self.assertEqual(self.request(component, self.op+'-reuse').outcome, 'reuse')
                self.assertEqual(provider.calls, [])

    def test_same_identity_resumes_only_remaining_batch(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.prepare_partial(component)
                provider = self.restart(component)
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'resuming'); self.finish()
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(self.service.status('book', self.op).state, 'succeeded')
                self.assertEqual(len(self.repo.read_checkpoints('book', self.op)), 2)
                self.assertIsNotNone(self.saved(component))

    def complete_partial(self, component, *, mixed):
        self.prepare_partial(component, previous=True)
        saved = self.saved(component)
        provider = self.restart(component, model='model-B' if mixed else None)
        settings, block, sentences = self.service.context.selection('book', self.block.source.content_block_id)
        prepared = self.service.context.prepare(component, settings, block, sentences)
        snapshot = self.repo.inspect_operation('book', self.op)
        dispatched = self.service.worker.write('book', snapshot, 'dispatch')
        batch = self.service.context.services[component].generate_batch(prepared, dispatched.operation, now_ms=self.now)
        self.service.worker.write('book', dispatched, 'checkpoint', batch_index=1, batch=batch)
        return provider, saved

    def test_complete_compatible_checkpoints_finish_offline_after_expiry(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                provider, saved = self.complete_partial(component, mixed=False)
                self.now += 10001
                with patch.object(provider, 'get_model_capabilities', side_effect=AssertionError('offline capability')), \
                     patch.object(provider, 'count_input_tokens', side_effect=AssertionError('offline tokenizer')), \
                     patch.object(provider, 'complete_text_with_output_budget', side_effect=AssertionError('offline generation')):
                    self.assertEqual(self.service.reconcile('book', self.op).outcome, 'resuming'); self.finish()
                self.assertEqual(self.service.status('book', self.op).state, 'succeeded')
                self.assertGreater(self.saved(component).context.result_revision, saved.context.result_revision)
                self.assertEqual(len(provider.calls), 1)  # Only the pre-existing second checkpoint.

    def test_complete_mixed_checkpoints_fail_offline_and_allow_full_retry(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                provider, saved = self.complete_partial(component, mixed=True)
                parts = self.repo.read_checkpoints('book', self.op)
                receipt = self.repo.lookup_command('book', self.op)
                self.now += 10001
                with patch.object(provider, 'get_model_capabilities', side_effect=AssertionError('offline capacity')):
                    self.service.reconcile('book', self.op); self.finish()
                failed = self.service.status('book', self.op)
                self.assertEqual((failed.state, failed.snapshot.operation.failure_code), ('failed', 'assembly_invalid'))
                self.assertEqual(self.saved(component), saved)
                self.assertEqual(self.repo.read_checkpoints('book', self.op), parts)
                self.assertEqual(self.repo.lookup_command('book', self.op), receipt)
                self.service = self.make_service()
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'failed')
                self.assertEqual(len(provider.calls), 1)
                retry = self.request(component, self.op+'-retry', intent='retry', retry_operation_id=self.op)
                self.assertEqual(retry.outcome, 'claimed'); self.finish()
                self.assertEqual(len(provider.calls), 3)
                self.assertEqual(self.service.status('book', retry.operation_id).state, 'succeeded')
                self.assertEqual(self.saved(component).context.provenance.model, 'model-B')

    def test_failure_ack_loss_is_unknown_until_durable_readback(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.prepare_partial(component)
                provider = self.restart(component, model='model-B')
                with self.fault_write('fail'):
                    response = self.service.reconcile('book', self.op)
                self.assertEqual(response.outcome, 'unknown')
                self.assertEqual(provider.calls, [])
                self.assertEqual(self.service.status('book', self.op).state, 'failed')
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'failed')

    def test_changed_source_remains_stale_without_dispatch(self):
        self.prepare_partial('translation')
        provider = self.restart('translation', model='model-B')
        self.change()
        self.assertEqual(self.service.reconcile('book', self.op).outcome, 'stale')
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.repo.inspect_operation('book', self.op).operation.failure_code, 'binding_changed')

    def test_prompt_identity_is_checked_before_dispatch(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.prepare_partial(component)
                provider = self.restart(component)
                service = self.service.context.services[component]
                provenance = replace(service.generation_provenance(self.prepared), prompt_version='other-prompt')
                with patch.object(service, 'generation_provenance', return_value=provenance):
                    self.service.reconcile('book', self.op)
                self.assertEqual(provider.calls, [])
                self.assertEqual(self.service.status('book', self.op).snapshot.operation.failure_code,
                                 'recovery_incompatible')

    def test_unconfirmed_failure_write_does_not_claim_failed(self):
        self.prepare_partial('translation')
        provider = self.restart('translation', model='model-B')
        before = self.repo.inspect_operation('book', self.op)
        with patch.object(self.repo, 'write_operation', side_effect=LearningStorageError('unavailable')):
            with self.assertRaises(LearningStorageError):
                self.service.reconcile('book', self.op)
        self.assertEqual(self.repo.inspect_operation('book', self.op), before)
        self.assertEqual(provider.calls, [])

    def test_expiry_winning_failure_cas_is_not_overwritten(self):
        self.prepare_partial('translation')
        provider = self.restart('translation', model='model-B')
        original = self.repo.write_operation
        def expire_first(document, command, **kwargs):
            self.now += 10001
            original(document, replace(command, action='expire', now_ms=self.now,
                                       reason=None, outcome_known=None), **kwargs)
            return original(document, command, **kwargs)
        with patch.object(self.repo, 'write_operation', side_effect=expire_first):
            with self.assertRaisesRegex(LearningContractError, 'revision_conflict'):
                self.service.reconcile('book', self.op)
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.service.status('book', self.op).snapshot.operation.failure_code, 'lease_expired')

    def test_changed_settings_remains_stale_not_assembly_failure(self):
        self.prepare_partial('translation')
        provider = self.restart('translation', model='model-B')
        self.settings(normal=True)
        self.assertEqual(self.service.reconcile('book', self.op).outcome, 'stale')
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.repo.inspect_operation('book', self.op).operation.failure_code, 'binding_changed')


class FileRecoveryTests(RecoveryCases, unittest.TestCase):
    setUp = FileLearningOrchestrationTests.setUp


class PostgresRecoveryTests(RecoveryCases, unittest.TestCase):
    setUpClass = classmethod(SqlLearningOrchestrationTests.setUpClass.__func__)
    drop_schema = classmethod(SqlLearningOrchestrationTests.drop_schema.__func__)
    setUp = SqlLearningOrchestrationTests.setUp


if __name__ == '__main__':
    unittest.main()
