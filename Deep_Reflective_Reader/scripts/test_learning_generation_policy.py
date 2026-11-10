"""LLRN-05 pure policy and simulated CAS races; NOT SQL/file acceptance."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
from threading import Barrier, Lock
import unittest

from language.language_proficiency_registry import ProficiencySelectionError
from section_tasks.learning_cache import HintsKey, TranslationKey, validate_complete_result
from section_tasks.learning_generation_policy import (AdmissionDecision, BatchPlan, CacheEntry,
    GenerationBinding, GenerationCommand, checkpoint_batch, commit_result, dispatch_batch,
    expire_operation, finish_failure, invalidate_operation, plan_request, read_entry, renew_lease)
from shared.learning_source import (LearningContractError, LearningSourceBinding, SentenceResource,
    SourceSpan, text_hash)
from shared.learning_results import (AlignedTranslation, GenerationProvenance, LearningHints,
    LearningProficiencyRef, LearningResultContext, TranslatedSentence)

TEXT = 'Hello! Welcome.'


class MemoryCAS:
    """Test-only transaction boundary. No production durability claim."""
    def __init__(self, entry, binding):
        self.entry, self.binding = entry, binding
        self.receipts, self.history = {}, {}
        self.lock = Lock()

    def apply(self, decision, binding):
        with self.lock:
            if self.binding != binding or self.entry.revision != decision.expected_entry_revision:
                return False
            receipt = getattr(decision, 'receipt', None)
            if receipt is not None:
                old = self.receipts.get(receipt.command.command_id)
                if old is not None:
                    return False
            if decision.next_entry is not None:
                self.entry = decision.next_entry
                op = self.entry.operation
                if op:
                    self.history[op.operation_id] = op
            if receipt is not None:
                self.receipts[receipt.command.command_id] = receipt
            return True


class LearningGenerationPolicyTests(unittest.TestCase):
    def setUp(self):
        source = LearningSourceBinding('workspace', 'book', 'chapter', 'section', 'unit',
            'unit:content:0', 1, 'doc-epoch', 'block-epoch', text_hash(TEXT))
        self.sentences = SentenceResource.from_spans(source=source, effective_text=TEXT,
            segmentation_version='v1', segmentation_language='en',
            spans=(SourceSpan(0, 6), SourceSpan(7, len(TEXT))))
        self.key = TranslationKey(self.sentences.reference, 'en', 'zh', 'policy-1', 'prompt-1')
        self.binding = GenerationBinding(self.key, 1, 2)
        self.entry = CacheEntry(self.key)
        self.plan = BatchPlan((tuple(s.sentence_id for s in self.sentences.sentences),))

    def hints_key(self, level='B1'):
        return HintsKey(self.sentences.reference, 'en', 'zh', 'hints-1', 'hints-prompt-1',
                        LearningProficiencyRef('cefr_reading', '1', level))

    def result(self, key=None, revision=1):
        key = key or self.key
        ctx = LearningResultContext(key.resource, key.source_language, key.explanation_language,
            key.policy_version, revision, GenerationProvenance('provider', 'model', key.prompt_version))
        if type(key) is HintsKey:
            return LearningHints(ctx, key.proficiency, ())
        return AlignedTranslation(ctx, tuple(TranslatedSentence(s.sentence_id, '譯文')
                                             for s in self.sentences.sentences))

    def request(self, entry=None, command=None, **overrides):
        entry = entry or self.entry
        binding = replace(self.binding, key=entry.key)
        params = dict(current=binding, sentences=self.sentences, effective_text=TEXT,
                      plan=self.plan, now_ms=10, lease_until_ms=100)
        params.update(overrides)
        return plan_request(entry, command or GenerationCommand('command-1', binding), **params)

    def args(self, entry, now=20, **extra):
        op = entry.operation
        return dict(operation_id=op.operation_id, fence=op.fence, now_ms=now, **extra)

    def claimed(self, entry=None, **kwargs):
        return self.request(entry, **kwargs).next_entry

    def running(self, entry=None):
        entry = entry or self.claimed()
        return dispatch_batch(entry, current=entry.operation.binding, **self.args(entry)).next_entry

    def partial(self, entry=None, index=0, now=30):
        entry = entry or self.running()
        return checkpoint_batch(entry, current=entry.operation.binding, batch_index=index,
            checkpoint_digest=text_hash('batch-' + str(index)), **self.args(entry, now)).next_entry

    def finish(self, entry=None, result=None, **overrides):
        entry = entry or self.partial()
        args = dict(current=entry.operation.binding, result=result or self.result(entry.key),
                    sentences=self.sentences, effective_text=TEXT, **self.args(entry, 40))
        args.update(overrides)
        return commit_result(entry, **args)

    def unknown(self, entry=None):
        entry = entry or self.running()
        return finish_failure(entry, reason='provider_outcome_unknown', outcome_known=False,
                              **self.args(entry, 30)).next_entry

    def test_translation_has_no_level_or_result_revision_or_model_dimension(self):
        self.assertEqual({f.name for f in fields(self.key)}, {'resource', 'source_language',
            'explanation_language', 'policy_version', 'prompt_version'})
        self.assertEqual(self.key.cache_id, replace(self.key).cache_id)
        saved = self.result(revision=7)
        changed = replace(saved, context=replace(saved.context,
            provenance=replace(saved.context.provenance, model='new-model')))
        validate_complete_result(self.key, changed, self.sentences, TEXT)

    def test_hints_level_scheme_version_and_component_separate(self):
        key = self.hints_key()
        keys = [key, self.hints_key('B2'), replace(key, proficiency=replace(key.proficiency,
            scheme_version='2')), replace(key, proficiency=replace(key.proficiency, scheme_id='future')),
            TranslationKey(key.resource, key.source_language, key.explanation_language,
                           key.policy_version, key.prompt_version)]
        self.assertEqual(len({k.cache_id for k in keys}), len(keys))
        self.assertNotIn('translation_revision', key.to_dict())

    def test_all_source_and_policy_dimensions_invalidate_identity(self):
        for name, value in dict(namespace='other', document_id='other', chapter_id='other',
            section_id='other', task_unit_id='other', content_block_id='other', structure_version=2,
            document_epoch='new', content_epoch='new', effective_content_hash=text_hash('changed')).items():
            changed_ref = replace(self.key.resource, source=replace(self.key.resource.source, **{name:value}))
            self.assertNotEqual(replace(self.key, resource=changed_ref).cache_id, self.key.cache_id, name)
        for change in (dict(source_language='fr'), dict(explanation_language='ja'),
                       dict(policy_version='2'), dict(prompt_version='2'),
                       dict(resource=replace(self.key.resource, resource_id=text_hash('other segmentation')))):
            self.assertNotEqual(replace(self.key, **change).cache_id, self.key.cache_id)

    def test_settings_revisions_are_guards_not_cache_identity(self):
        other = replace(self.binding, book_revision=3, workspace_revision=2)
        self.assertEqual(other.key, self.binding.key)
        with self.assertRaisesRegex(LearningContractError, 'binding_stale'):
            self.request(current=other)
        saved = CacheEntry(self.key, saved=self.result())
        decision = self.request(saved, GenerationCommand('new', other), current=other, plan=None)
        self.assertEqual(decision.outcome, 'reuse')

    def test_strict_registry_and_same_language_guard(self):
        for code in ('EN', 'unknown', 'bogus'):
            key = replace(self.key, source_language=code)
            with self.assertRaisesRegex(LearningContractError, 'language_invalid'):
                self.request(CacheEntry(key))
        with self.assertRaisesRegex(LearningContractError, 'languages_equal'):
            replace(self.key, explanation_language='en')
        with self.assertRaises(ProficiencySelectionError):
            self.request(CacheEntry(self.hints_key('invalid')))

    def test_read_only_missing_and_error_states_never_start_work(self):
        self.assertEqual(read_entry(self.entry).state, 'missing')
        for evidence in ('read_error', 'unsupported', 'stale'):
            self.assertEqual(read_entry(self.entry, evidence=evidence).state, evidence)
            with self.assertRaisesRegex(LearningContractError, 'read_not_verified'):
                self.request(evidence=evidence)
        self.assertEqual(self.entry, CacheEntry(self.key))

    def test_saved_empty_hints_is_ready_and_reused_without_admission(self):
        key = self.hints_key()
        entry = CacheEntry(key, saved=self.result(key))
        class NoCalls:
            def evaluate(self, command):
                raise AssertionError('cache reads must bypass admission')
        self.assertEqual(read_entry(entry).state, 'ready')
        self.assertEqual(self.request(entry, admission=NoCalls(), plan=None).outcome, 'reuse')

    def test_claim_and_dispatch_are_distinct_commits(self):
        entry = self.claimed()
        self.assertEqual(entry.operation.state, 'claimed')
        self.assertEqual(self.entry.operation, None)
        running = self.running(entry)
        self.assertTrue(running.operation.outcome_uncertain)
        with self.assertRaisesRegex(LearningContractError, 'dispatch_forbidden'):
            self.running(running)

    def test_same_command_replay_has_no_new_claim(self):
        decision = self.request()
        replay = self.request(decision.next_entry, receipt=decision.receipt)
        self.assertTrue(replay.replay)
        self.assertIsNone(replay.next_entry)
        self.assertEqual(replay.receipt, decision.receipt)

    def test_same_command_changed_intent_key_or_settings_conflicts(self):
        receipt = self.request().receipt
        commands = [replace(receipt.command, intent='regenerate'),
                    replace(receipt.command, binding=replace(self.binding, book_revision=3)),
                    replace(receipt.command, binding=replace(self.binding, key=self.hints_key()))]
        for command in commands:
            with self.assertRaisesRegex(LearningContractError, 'command_(conflict|key_mismatch)'):
                self.request(command=command, receipt=receipt)

    def test_different_commands_same_key_coalesce(self):
        entry = self.claimed()
        other = GenerationCommand('other', self.binding, 'regenerate')
        decision = self.request(entry, other, plan=None)
        self.assertEqual(decision.outcome, 'coalesced')
        self.assertEqual(decision.receipt.operation_id, entry.operation.operation_id)
        self.assertIsNone(decision.next_entry)

    def test_parallel_claims_and_dispatch_use_cas(self):
        store = MemoryCAS(self.entry, self.binding)
        barrier = Barrier(12)
        def contender(index):
            command = GenerationCommand('command-' + str(index), self.binding)
            decision = self.request(store.entry, command)
            barrier.wait()
            while not store.apply(decision, self.binding):
                decision = self.request(store.entry, command)
            return decision.outcome
        with ThreadPoolExecutor(max_workers=12) as pool:
            outcomes = list(pool.map(contender, range(12)))
        self.assertEqual(outcomes.count('claimed'), 1)
        self.assertEqual(outcomes.count('coalesced'), 11)
        a = dispatch_batch(store.entry, current=self.binding, **self.args(store.entry))
        b = dispatch_batch(store.entry, current=self.binding, **self.args(store.entry))
        calls = sum(store.apply(d, self.binding) for d in (a,b))
        self.assertEqual(calls, 1)
        self.assertEqual(len(store.receipts), 12)

    def test_failed_refresh_preserves_saved_result_and_is_not_missing(self):
        saved = self.result()
        entry = CacheEntry(self.key, saved=saved)
        command = GenerationCommand('refresh', self.binding, 'regenerate')
        running = self.running(self.claimed(entry, command=command))
        failed = finish_failure(running, reason='provider_rejected', outcome_known=True,
                                **self.args(running, 30)).next_entry
        self.assertEqual(failed.saved, saved)
        self.assertEqual((read_entry(failed).state, read_entry(failed).operation_state), ('ready','failed'))
        self.assertEqual(self.request(failed).outcome, 'reuse')
        self.assertEqual(self.request(failed, replace(command, command_id='refresh-2')).outcome, 'retry_required')

    def test_refresh_replaces_only_on_success_and_preserves_other_component(self):
        old = self.result()
        hints = CacheEntry(self.hints_key(), saved=self.result(self.hints_key()))
        refresh = self.claimed(CacheEntry(self.key, saved=old),
            command=GenerationCommand('refresh', self.binding, 'regenerate'))
        partial = self.partial(self.running(refresh))
        self.assertEqual(partial.saved, old)
        completed = self.finish(partial, self.result(revision=2)).next_entry
        self.assertEqual(completed.result_revision, 2)
        self.assertEqual(completed.operation.result_revision, 2)
        self.assertEqual(completed.operation.state, 'succeeded')
        self.assertEqual(hints.result_revision, 1)

    def test_failure_without_saved_result_requires_explicit_retry(self):
        entry = self.running()
        failed = finish_failure(entry, reason='invalid_output', outcome_known=True,
                                **self.args(entry, 30)).next_entry
        self.assertEqual(read_entry(failed).state, 'failed')
        self.assertEqual(self.request(failed).outcome, 'retry_required')
        cmd = GenerationCommand('retry', self.binding, 'retry', failed.operation.operation_id)
        retry = self.request(failed, cmd, now_ms=40).next_entry
        self.assertGreater(retry.last_fence, failed.last_fence)

    def test_unknown_never_auto_retries_or_blindly_regenerates(self):
        unknown = self.unknown()
        for intent in ('get_or_generate', 'regenerate'):
            cmd = GenerationCommand('other', self.binding, intent)
            self.assertEqual(self.request(unknown, cmd, now_ms=40).outcome, 'reconcile')
        retry = GenerationCommand('retry', self.binding, 'retry', unknown.operation.operation_id)
        self.assertEqual(self.request(unknown, retry, now_ms=40).outcome, 'reconcile')
        with self.assertRaisesRegex(LearningContractError, 'dispatch_forbidden'):
            dispatch_batch(unknown, current=self.binding, **self.args(unknown, 40))

    def test_manual_unknown_retry_acknowledges_cost_and_fences_late_writer(self):
        unknown = self.unknown()
        cmd = GenerationCommand('retry', self.binding, 'retry', unknown.operation.operation_id, True)
        new = self.request(unknown, cmd, now_ms=40).next_entry
        self.assertGreater(new.last_fence, unknown.last_fence)
        with self.assertRaisesRegex(LearningContractError, 'fence_conflict'):
            checkpoint_batch(new, current=self.binding, batch_index=0, checkpoint_digest=text_hash('x'),
                              **self.args(unknown, 50))

    def test_reconcile_late_response_without_provider_resend(self):
        unknown = self.unknown()
        reconciled = self.partial(unknown, now=40)
        completed = self.finish(reconciled, now_ms=50).next_entry
        self.assertEqual(completed.operation.state, 'succeeded')
        self.assertFalse(completed.operation.outcome_uncertain)

    def test_expiry_before_dispatch_is_failed_after_dispatch_is_unknown(self):
        for entry, expected in ((self.claimed(), 'failed'), (self.running(), 'unknown'), (self.partial(), 'failed')):
            expired = expire_operation(entry, **self.args(entry, 100)).next_entry
            self.assertEqual(expired.operation.state, expected)
            self.assertEqual(self.request(entry, now_ms=100, lease_until_ms=200).outcome, 'reconcile')
        with self.assertRaisesRegex(LearningContractError, 'expiry_forbidden'):
            entry = self.running()
            expire_operation(entry, **self.args(entry, 99))

    def test_lease_renewal_requires_live_owner_and_never_resurrects(self):
        entry = self.running()
        renewed = renew_lease(entry, current=self.binding, lease_until_ms=200,
                              **self.args(entry, 99)).next_entry
        self.assertEqual(renewed.operation.lease_until_ms, 200)
        self.assertEqual(renewed.operation.state, 'running')
        with self.assertRaisesRegex(LearningContractError, 'renewal_forbidden'):
            renew_lease(entry, current=self.binding, lease_until_ms=200, **self.args(entry, 100))

    def test_settings_change_stops_dispatch_and_commit_and_adapter_cas(self):
        current = replace(self.binding, book_revision=3)
        entry = self.claimed()
        with self.assertRaisesRegex(LearningContractError, 'binding_stale'):
            dispatch_batch(entry, current=current, **self.args(entry))
        with self.assertRaisesRegex(LearningContractError, 'binding_stale'):
            self.finish(current=current)
        store = MemoryCAS(entry, current)
        proposed = dispatch_batch(entry, current=self.binding, **self.args(entry))
        self.assertFalse(store.apply(proposed, self.binding))

    def test_staling_running_operation_does_not_erase_unknown_cost(self):
        entry = self.running()
        current = replace(self.binding, book_revision=3)
        stale = invalidate_operation(entry, current=current, **self.args(entry, 30)).next_entry
        self.assertTrue(stale.operation.outcome_uncertain)
        cmd = GenerationCommand('retry', current, 'retry', entry.operation.operation_id)
        self.assertEqual(self.request(stale, cmd, current=current, now_ms=40).outcome, 'reconcile')
        cmd = replace(cmd, acknowledge_possible_duplicate_cost=True)
        self.assertEqual(self.request(stale, cmd, current=current, now_ms=40).outcome, 'claimed')

    def test_source_version_mismatch_rejected_before_claim_and_commit(self):
        # Build new deterministic IDs through the public factory.
        source = replace(self.sentences.source, content_epoch='new-epoch')
        changed = SentenceResource.from_spans(source=source, effective_text=TEXT,
            segmentation_version='v2', segmentation_language='en',
            spans=tuple(s.span for s in self.sentences.sentences))
        with self.assertRaises(LearningContractError):
            self.request(sentences=changed)
        with self.assertRaises(LearningContractError):
            self.finish(sentences=changed)
        with self.assertRaises(LearningContractError):
            self.finish(effective_text='changed')

    def test_result_key_revision_alignment_and_component_validation(self):
        result = self.result()
        bad = [replace(result, rows=result.rows[:1]), replace(result, rows=result.rows[::-1]),
               replace(result, context=replace(result.context, result_revision=2)),
               replace(result, context=replace(result.context, policy_version='other')),
               replace(result, context=replace(result.context,
                   provenance=replace(result.context.provenance, prompt_version='other'))),
               self.result(self.hints_key())]
        for item in bad:
            with self.assertRaises(LearningContractError):
                self.finish(result=item)

    def test_result_revision_cas_rejects_intervening_writer(self):
        partial = self.partial()
        competing = replace(partial, saved=self.result())
        with self.assertRaisesRegex(LearningContractError, 'result_cas_conflict'):
            self.finish(competing, self.result(revision=2))

    def test_batch_partition_requires_exact_order_coverage_without_duplicates(self):
        ids = self.plan.groups[0]
        for groups in ((ids[::-1],), (ids[:1],), (('foreign',),)):
            with self.assertRaisesRegex(LearningContractError, 'coverage_invalid'):
                self.request(plan=BatchPlan(groups))
        for groups in ((), ((),), ((ids[0],),(ids[0],))):
            with self.assertRaises(LearningContractError):
                BatchPlan(groups)

    def test_multibatch_partial_is_not_ready_and_cannot_commit_early(self):
        ids = self.plan.groups[0]
        entry = self.claimed(plan=BatchPlan(((ids[0],),(ids[1],))))
        partial = self.partial(self.running(entry))
        self.assertEqual(read_entry(partial).state, 'partial')
        with self.assertRaisesRegex(LearningContractError, 'result_incomplete'):
            self.finish(partial)
        running = dispatch_batch(partial, current=self.binding, **self.args(partial, 40)).next_entry
        with self.assertRaisesRegex(LearningContractError, 'checkpoint_forbidden'):
            self.partial(running, index=0, now=50)
        complete = self.partial(running, index=1, now=50)
        self.assertEqual(self.finish(complete, now_ms=60).next_entry.operation.state, 'succeeded')

    def test_unknown_batch_recovery_retains_prior_progress(self):
        ids = self.plan.groups[0]
        partial = self.partial(self.running(self.claimed(plan=BatchPlan(((ids[0],),(ids[1],))))))
        running = dispatch_batch(partial, current=self.binding, **self.args(partial, 40)).next_entry
        unknown = expire_operation(running, **self.args(running, 100)).next_entry
        self.assertEqual(unknown.operation.checkpoints, partial.operation.checkpoints)
        recovered = self.partial(unknown, index=1, now=110)
        self.assertEqual(self.finish(recovered, now_ms=120).next_entry.operation.state, 'succeeded')

    def test_admission_denial_does_not_modify_cache_or_claim(self):
        class Deny:
            def evaluate(self, command):
                return AdmissionDecision(False, 'future_restriction')
        decision = self.request(admission=Deny())
        self.assertEqual(decision.outcome, 'denied')
        self.assertEqual(decision.receipt.admission_reason, 'future_restriction')
        self.assertIsNone(decision.next_entry)
        self.assertEqual(self.entry.revision, 0)

    def test_receipt_history_replays_old_command_after_new_refresh(self):
        first = self.request()
        done = self.finish(self.partial(self.running(first.next_entry))).next_entry
        refreshed = self.request(done, GenerationCommand('refresh', self.binding, 'regenerate'), now_ms=50)
        replay = self.request(refreshed.next_entry, receipt=first.receipt)
        self.assertTrue(replay.replay)
        self.assertEqual(replay.receipt.operation_id, first.receipt.operation_id)
        self.assertIsNone(replay.next_entry)

    def test_inactive_mode_and_invalid_retry_target_rejected(self):
        current = replace(self.binding, active_mode='normal')
        with self.assertRaisesRegex(LearningContractError, 'binding_stale'):
            self.request(command=GenerationCommand('x',current), current=current)
        with self.assertRaisesRegex(LearningContractError, 'retry_target_invalid'):
            self.request(command=GenerationCommand('x',self.binding,'retry','missing'))

    def test_worker_stop_during_dispatch_cannot_claim_known_failure(self):
        entry = self.running()
        with self.assertRaisesRegex(LearningContractError, 'failure_evidence_invalid'):
            finish_failure(entry, reason='worker_stopped', outcome_known=True, **self.args(entry,30))
        unknown = self.unknown(entry)
        failed = finish_failure(unknown, reason='provider_rejected', outcome_known=True,
                                **self.args(unknown,40)).next_entry
        self.assertEqual(failed.operation.state, 'failed')
        self.assertFalse(failed.operation.outcome_uncertain)

    def test_empty_source_never_claims_provider_work(self):
        text = '  '
        source = replace(self.sentences.source, effective_content_hash=text_hash(text))
        sentences = SentenceResource.from_spans(source=source, effective_text=text,
            segmentation_version='v1', segmentation_language='en', spans=())
        key = replace(self.key, resource=sentences.reference)
        result = self.request(CacheEntry(key), sentences=sentences, effective_text=text, plan=None)
        self.assertEqual(result.outcome, 'no_content')
        self.assertIsNone(result.next_entry)

    def test_lost_commit_ack_reconciles_receipt_and_saved_result(self):
        first = self.request()
        store = MemoryCAS(self.entry, self.binding)
        self.assertTrue(store.apply(first, self.binding))
        dispatch = dispatch_batch(store.entry, current=self.binding, **self.args(store.entry))
        self.assertTrue(store.apply(dispatch, self.binding))
        checkpoint = checkpoint_batch(store.entry, current=self.binding, batch_index=0,
            checkpoint_digest=text_hash('complete'), **self.args(store.entry,30))
        self.assertTrue(store.apply(checkpoint, self.binding))
        commit = self.finish(store.entry)
        self.assertTrue(store.apply(commit, self.binding))  # acknowledgement gets lost
        self.assertFalse(store.apply(commit, self.binding))  # no duplicate commit
        replay = self.request(store.entry, receipt=store.receipts['command-1'])
        self.assertTrue(replay.replay)
        self.assertEqual(store.history[replay.receipt.operation_id].state, 'succeeded')
        self.assertEqual(read_entry(store.entry).state, 'ready')

    def test_failed_commit_keeps_old_result_and_pending_operation(self):
        old = self.result()
        refresh = self.claimed(CacheEntry(self.key, saved=old),
            command=GenerationCommand('refresh',self.binding,'regenerate'))
        entry = self.partial(self.running(refresh))
        store = MemoryCAS(entry, replace(self.binding, workspace_revision=2))
        proposed = self.finish(entry, self.result(revision=2))
        self.assertFalse(store.apply(proposed,self.binding))
        self.assertEqual(store.entry.saved,old)
        self.assertEqual(store.entry.operation.state,'partial')

    def test_pending_unknown_and_success_receipt_without_output_are_not_missing(self):
        for entry in (self.claimed(),self.running(),self.partial(),self.unknown()):
            self.assertNotEqual(read_entry(entry).state,'missing')
        completed = self.finish().next_entry
        damaged = replace(completed,saved=None)
        self.assertEqual(read_entry(damaged).state,'unavailable')
        self.assertEqual(self.request(damaged).outcome,'unavailable')
        with self.assertRaisesRegex(LearningContractError,'operation_missing'):
            replace(completed,operation=None)

    def test_read_during_unknown_refresh_retains_ready_data(self):
        old = self.result()
        entry = self.claimed(CacheEntry(self.key,saved=old),
            command=GenerationCommand('refresh',self.binding,'regenerate'))
        unknown = self.unknown(self.running(entry))
        view = read_entry(unknown)
        self.assertEqual((view.state,view.operation_state),('ready','unknown'))
        self.assertEqual(view.saved,old)
        self.assertEqual(self.request(unknown).outcome,'reuse')

    def test_immutable_contracts_and_strict_integer_validation(self):
        with self.assertRaises(FrozenInstanceError):
            self.entry.revision = 1
        for change in (dict(revision=True), dict(last_fence=-1)):
            with self.assertRaises(LearningContractError):
                replace(self.entry, **change)
        with self.assertRaises(LearningContractError):
            self.request(now_ms=True)


if __name__ == '__main__':
    unittest.main()
