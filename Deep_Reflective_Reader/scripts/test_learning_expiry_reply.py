"""LEARN-02: actual file/SQL publication with deterministic dispatch/expiry races."""
from dataclasses import replace
from threading import Event
import unittest
from unittest.mock import patch

from scripts.test_learning_orchestration import (
    OrchestrationCases, FileLearningOrchestrationTests, SqlLearningOrchestrationTests,
)
from section_tasks.learning_translation_service import _prompt
from section_tasks.learning_hints_prompt import build_prompt
from document_structure.learning_repository import LearningCommitUnknown


class ExpiryReplyCases:
    setup_service = OrchestrationCases.setup_service
    make_service = OrchestrationCases.make_service
    request = OrchestrationCases.request
    finish = OrchestrationCases.finish
    read = OrchestrationCases.read
    change = OrchestrationCases.change
    settings = OrchestrationCases.settings
    fault_write = OrchestrationCases.fault_write

    def begin(self, component='translation', *, multi=False, observe=True):
        self.service = self.make_service()
        self.op = component + '-expiry'
        self.provider = self.trans_provider if component == 'translation' else self.hints_provider
        if multi:
            settings, block, sentences = self.service.context.selection('book', self.block.source.content_block_id)
            prepared = self.service.context.prepare(component, settings, block, sentences)
            prompt = _prompt if component == 'translation' else build_prompt
            limit = max(len(prompt(prepared.key, sentences, block.effective_text, (s.sentence_id,)))
                        for s in sentences.sentences) + 256
            self.provider.caps = replace(self.provider.caps, max_input_tokens=limit)
        entered, self.release = Event(), Event()
        original = self.provider.complete_text_with_output_budget
        def blocking(prompt, budget):
            entered.set()
            if not self.release.wait(20):
                raise TimeoutError('test response not released')
            return original(prompt, budget)
        self.provider.complete_text_with_output_budget = blocking
        self.addCleanup(self.release.set)
        self.assertEqual(self.request(component, self.op).outcome, 'claimed')
        self.assertTrue(entered.wait(10))
        self.dispatched = self.repo.inspect_operation('book', self.op)
        self.assertEqual(self.dispatched.operation.state, 'running')
        self.assertEqual(len(self.dispatched.operation.plan.groups), 2 if multi else 1)
        self.now += 10001
        if observe:
            for _ in range(3):
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'unknown')
            current = self.repo.inspect_operation('book', self.op)
            self.assertEqual(current.entry_revision, self.dispatched.entry_revision + 1)
        return self.dispatched

    def done(self):
        self.release.set(); self.finish()
        return self.repo.inspect_operation('book', self.op)

    def test_observed_expiry_preserves_reply_exactly_once(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.begin(component)
                committed = []
                original = self.repo.write_operation
                def record(doc, command, **kwargs):
                    result = original(doc, command, **kwargs)
                    if command.action == 'checkpoint': committed.append(command)
                    return result
                with patch.object(self.repo, 'write_operation', side_effect=record):
                    current = self.done()
                self.assertEqual(current.operation.state, 'succeeded')
                self.assertEqual(len(self.provider.calls), 1)
                self.assertEqual(len(committed), 1)
                self.assertEqual(committed[0].expected_entry_revision, self.dispatched.entry_revision + 1)
                self.assertEqual(len(self.repo.read_checkpoints('book', self.op)), 1)
                saved = self.read().blocks[0].components[0 if component == 'translation' else 1].view.saved
                self.assertIsNotNone(saved)
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'succeeded')
                self.assertEqual(len(self.provider.calls), 1)

    def test_expiry_without_observation_control(self):
        self.begin(observe=False)
        self.assertEqual(self.done().operation.state, 'succeeded')
        self.assertEqual(len(self.provider.calls), 1)

    def test_expired_multi_batch_saves_reply_without_dispatching_next(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.begin(component, multi=True)
                current = self.done()
                self.assertEqual((current.operation.state, current.operation.failure_code), ('failed', 'lease_expired'))
                self.assertEqual(len(current.operation.checkpoints), 1)
                self.assertEqual(len(self.provider.calls), 1)
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'failed')
                self.assertEqual(len(self.provider.calls), 1)

    def test_source_change_rejects_recovered_reply(self):
        self.begin(); self.change()
        current = self.done()
        self.assertEqual(current.operation.checkpoints, ())
        self.assertIsNone(self.read().blocks[0].components[0].view.saved)
        self.assertEqual(len(self.provider.calls), 1)

    def test_settings_change_rejects_recovered_reply(self):
        self.begin(); self.settings(level='B2')
        self.assertEqual(self.done().operation.checkpoints, ())
        self.assertIsNone(self.read().blocks[0].components[0].view.saved)
        self.assertEqual(len(self.provider.calls), 1)

    def test_new_fence_rejects_original_worker(self):
        self.begin()
        newer = self.make_service()
        denied = self.request(command_id='without-consent', service=newer, intent='retry', retry_operation_id=self.op)
        self.assertEqual(denied.outcome, 'reconcile')
        retry = self.request(command_id='new-owner', service=newer, intent='retry', retry_operation_id=self.op,
                             acknowledge_possible_duplicate_cost=True)
        self.assertEqual(retry.outcome, 'claimed')
        self.done(); self.finish(newer)
        self.assertEqual(self.repo.read_checkpoints('book', self.op), ())
        current = self.repo.inspect_operation('book', 'new-owner')
        self.assertGreater(current.operation.fence, self.dispatched.operation.fence)
        self.assertEqual(current.operation.state, 'succeeded')
        self.assertEqual(len(self.provider.calls), 2)  # One old call, one explicitly authorized retry.

    def test_occupied_batch_is_never_overwritten(self):
        for component, different in (('translation', True), ('hints', False)):
            with self.subTest(component=component, different=different):
                self.begin(component)
                original = self.repo.write_operation
                winner = []
                def compete(doc, command, **kwargs):
                    if command.action == 'checkpoint' and not winner:
                        current = self.repo.inspect_operation(doc, self.op)
                        batch = command.batch
                        if different:
                            batch = replace(batch, provenance=replace(batch.provenance, model='competing-model'))
                        original(doc, replace(command, expected_entry_revision=current.entry_revision, batch=batch), **kwargs)
                        winner.append(batch)
                    return original(doc, command, **kwargs)
                with patch.object(self.repo, 'write_operation', side_effect=compete):
                    current = self.done()
                self.assertEqual(current.operation.state, 'partial')
                self.assertEqual(self.repo.read_checkpoints('book', self.op), tuple(winner))
                self.assertEqual(len(self.provider.calls), 1)

    def test_source_change_after_readback_is_revalidated_under_write_lock(self):
        self.begin()
        original = self.repo.inspect_operation
        reads = []
        def race(*args, **kwargs):
            result = original(*args, **kwargs)
            if not reads:
                reads.append(result); self.change()
            return result
        with patch.object(self.repo, 'inspect_operation', side_effect=race):
            self.release.set(); self.finish()
        self.assertEqual(len(reads), 1)
        self.assertEqual(original('book', self.op).operation.checkpoints, ())
        self.assertEqual(len(self.provider.calls), 1)

    def test_checkpoint_ack_loss_stops_then_explicit_reconcile_finishes_offline(self):
        for component in ('translation', 'hints'):
            with self.subTest(component=component):
                self.begin(component)
                with self.fault_write('checkpoint'):
                    current = self.done()
                self.assertEqual(current.operation.state, 'partial')
                self.assertEqual(len(current.operation.checkpoints), 1)
                self.assertEqual(len(self.provider.calls), 1)
                self.service = self.make_service()
                self.assertEqual(self.service.reconcile('book', self.op).outcome, 'resuming'); self.finish()
                self.assertEqual(self.repo.inspect_operation('book', self.op).operation.state, 'succeeded')
                self.assertEqual(len(self.provider.calls), 1)

    def test_competing_checkpoint_after_readback_wins_second_cas(self):
        self.begin()
        original_write = self.repo.write_operation
        original_read = self.repo.inspect_operation
        attempted, winner = [], []
        def record(doc, command, **kwargs):
            if command.action == 'checkpoint': attempted.append(command)
            return original_write(doc, command, **kwargs)
        def compete(doc, operation_id):
            current = original_read(doc, operation_id)
            command = attempted[0]
            batch = replace(command.batch, provenance=replace(command.batch.provenance, model='winner'))
            original_write(doc, replace(command, expected_entry_revision=current.entry_revision, batch=batch),
                           validate_binding=self.service.context.validate_binding)
            winner.append(batch)
            return current
        with patch.object(self.repo, 'write_operation', side_effect=record), \
             patch.object(self.repo, 'inspect_operation', side_effect=compete):
            self.release.set(); self.finish()
        self.assertEqual(len(attempted), 2)
        self.assertEqual(len(winner), 1)
        self.assertEqual(self.repo.read_checkpoints('book', self.op), tuple(winner))
        self.assertEqual(original_read('book', self.op).operation.state, 'partial')
        self.assertEqual(len(self.provider.calls), 1)

    def test_unknown_second_checkpoint_commit_is_not_repeated(self):
        self.begin()
        original = self.repo.write_operation
        attempts = []
        def unknown(doc, command, **kwargs):
            if command.action == 'checkpoint':
                attempts.append(command)
                if len(attempts) == 2:
                    raise LearningCommitUnknown('unknown recovered write')
            return original(doc, command, **kwargs)
        with patch.object(self.repo, 'write_operation', side_effect=unknown):
            current = self.done()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(current.operation.state, 'unknown')
        self.assertEqual(current.operation.checkpoints, ())
        self.assertEqual(len(self.provider.calls), 1)

    def test_non_expiry_unknown_is_not_adopted_as_expiry(self):
        self.begin(observe=False)
        self.service.worker.write('book', self.dispatched, 'fail',
                                  reason='provider_outcome_unknown', outcome_known=False)
        current = self.done()
        self.assertEqual(current.operation.failure_code, 'provider_outcome_unknown')
        self.assertEqual(current.operation.checkpoints, ())
        self.assertEqual(len(self.provider.calls), 1)

    def test_unknown_checkpoint_outcome_never_retries_the_write_or_provider(self):
        self.begin()
        original = self.repo.write_operation
        attempts = []
        def unknown(doc, command, **kwargs):
            if command.action == 'checkpoint':
                attempts.append(command)
                raise LearningCommitUnknown('unknown commit result')
            return original(doc, command, **kwargs)
        with patch.object(self.repo, 'write_operation', side_effect=unknown):
            current = self.done()
        self.assertEqual(current.operation.state, 'unknown')
        self.assertEqual(current.operation.checkpoints, ())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.service.reconcile('book', self.op).outcome, 'unknown')


class FileExpiryReplyTests(ExpiryReplyCases, unittest.TestCase):
    setUp = FileLearningOrchestrationTests.setUp


class PostgresExpiryReplyTests(ExpiryReplyCases, unittest.TestCase):
    setUpClass = classmethod(SqlLearningOrchestrationTests.setUpClass.__func__)
    drop_schema = classmethod(SqlLearningOrchestrationTests.drop_schema.__func__)
    setUp = SqlLearningOrchestrationTests.setUp


if __name__ == '__main__':
    unittest.main()
