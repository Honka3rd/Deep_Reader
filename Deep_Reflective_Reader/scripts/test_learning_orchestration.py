"""LLRN-13 actual file/SQL orchestration, background ownership and paid-call fences."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import asyncio
from pathlib import Path
from threading import Barrier, Event
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning import LearningService, LearningBusy
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError
from document_structure.learning_file_lifecycle import FileLearningDocumentLifecycle
from document_structure.reading_content import ReadingContentService
from reading_preferences.models import BookSettings, LanguageLearningSettings
from reading_preferences.profile_source import ExistingProfileLanguageReader
from reading_preferences.service import ReadingPreferencesService
from scripts import test_learning_file_repository as files
from scripts import test_learning_postgres_repository as sql
from scripts.test_learning_translation_service import Provider as TranslationProvider
from scripts.test_learning_hints_service import Provider as HintsProvider
from section_tasks.learning_generation_policy import AdmissionDecision, BatchPlan
from section_tasks.learning_translation_service import LearningTranslationService
from section_tasks.learning_hints_service import LearningHintsService
from shared.learning_source import LearningContractError


class MutablePolicy:
    def __init__(self): self.allowed=True; self.calls=0
    def evaluate(self, command):
        self.calls+=1
        return AdmissionDecision(self.allowed,'allowed' if self.allowed else 'test_restricted')


class OrchestrationCases:
    def setup_service(self):
        self.profile=SimpleNamespace(document_language_code='en')
        self.now=1000
        self.trans_provider=TranslationProvider();self.hints_provider=HintsProvider()
        self.translation=LearningTranslationService(self.trans_provider,provider_name='test')
        self.hints=LearningHintsService(self.hints_provider,provider_name='test')
        self.preferences=ReadingPreferencesService(self.prefs,ExistingProfileLanguageReader(lambda scope,doc:self.profile))
        self.service=self.make_service()

    def make_service(self,repo=None,**kw):
        service=LearningService(repo or self.repo,self.preferences,self.translation,self.hints,
            lease_ms=10000,clock=lambda:self.now,**kw)
        self.addCleanup(service.shutdown)
        return service

    def request(self,component='translation',command_id='generate',service=None,**kw):
        current=self.preferences.read_book('book')
        block=self.repo.resolve_blocks('book',(self.block.source.content_block_id,))[0]
        return (service or self.service).request('book',block.source.content_block_id,component,
            command_id=command_id,expected_source=block.source,expected_workspace_revision=current.workspace.revision,
            expected_book_revision=current.book.revision,**kw)

    def finish(self,service=None):
        (service or self.service).shutdown()

    def change(self,text='Changed source.'):
        reader=ReadingContentService(self.docs);state=reader.read('book')
        return reader.mutate('book',action='edit',expected_revision=state['revision'],source_token=state['source_token'],
            mutation_id='edit-'+str(state['revision']),block_id=self.block.source.content_block_id,text=text,discard_interactions=False)

    def settings(self,*,level=None,normal=False):
        current=self.prefs.read_book('book');workspace=self.prefs.read_workspace()
        modes=(LanguageLearningSettings('en','cefr_reading','1',level),) if level else current.settings.modes
        settings=replace(current.settings,modes=modes,active_mode='normal' if normal else 'language_learning')
        self.prefs.save_book('book',settings,expected_revision=current.revision,
            expected_workspace_revision=workspace.revision,resolved_source_language='en',mutation_id='setting-'+str(current.revision))

    def read(self,service=None):
        return (service or self.service).read('book',(self.block.source.content_block_id,))

    def block_provider(self):
        entered,release=Event(),Event();original=self.trans_provider.complete_text_with_output_budget
        def blocking(prompt,budget):
            entered.set()
            if not release.wait(10): raise TimeoutError('test worker was not released')
            return original(prompt,budget)
        self.trans_provider.complete_text_with_output_budget=blocking
        self.addCleanup(release.set)
        return entered,release

    def test_wait_notifies_multiple_observers_without_status_polling(self):
        entered, release = self.block_provider()
        self.request()
        self.assertTrue(entered.wait(5))
        async def observe():
            with patch.object(self.service, 'status', wraps=self.service.status) as reads:
                one = asyncio.create_task(self.service.wait_for_completion('book', 'generate'))
                two = asyncio.create_task(self.service.wait_for_completion('book', 'generate'))
                for _ in range(1000):
                    if reads.call_count == 2:
                        break
                    await asyncio.sleep(.001)
                self.assertEqual(reads.call_count, 2)
                await asyncio.sleep(.03)
                self.assertFalse(one.done()); self.assertFalse(two.done())
                self.assertEqual(reads.call_count, 2)
                release.set()
                values = await asyncio.wait_for(asyncio.gather(one, two), 5)
                self.assertEqual([v.state for v in values], ['succeeded', 'succeeded'])
                self.assertEqual(reads.call_count, 4)
        asyncio.run(observe())
        self.assertFalse(self.service.worker.completion._listeners)
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_wait_timeout_cancel_and_completed_race_never_cancel_worker(self):
        entered, release = self.block_provider()
        self.request(); self.assertTrue(entered.wait(5))
        async def observe():
            with patch.object(self.service, 'status', wraps=self.service.status) as reads:
                pending = await self.service.wait_for_completion('book', 'generate', timeout=.01)
                self.assertEqual(pending.state, 'running')
                self.assertEqual(reads.call_count, 2)
            task = asyncio.create_task(self.service.wait_for_completion('book', 'generate'))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.service.worker.completion._listeners)
            release.set()
            done = await self.service.wait_for_completion('book', 'generate')
            self.assertEqual(done.state, 'succeeded')
            with patch.object(self.service, 'status', wraps=self.service.status) as reads:
                self.assertEqual((await self.service.wait_for_completion('book', 'generate')).state, 'succeeded')
                self.assertEqual(reads.call_count, 1)
                self.assertEqual((await self.service.wait_for_completion('book', 'absent')).state, 'missing')
        asyncio.run(observe())
        self.assertEqual(len(self.trans_provider.calls), 1)

    def test_wait_handles_completion_between_subscription_and_snapshot(self):
        self.request(); self.finish()
        self.assertEqual(asyncio.run(self.service.wait_for_completion('book', 'generate')).state, 'succeeded')
        self.assertFalse(self.service.worker.completion._listeners)

    def test_filtered_hints_report_persists_and_is_reused(self):
        from scripts.test_learning_hints_service import proposal, response_data
        settings, block, sentences = self.service.context.selection('book', self.block.source.content_block_id)
        prepared = self.service.context.prepare('hints', settings, block, sentences)
        data = response_data(prepared)
        sid = prepared.plan.groups[0][0]
        data['candidates'] = [proposal(sid, 'Hello'), proposal(sid, 'absent'),
                              proposal(sid, 'also absent', proper_name=True)]
        self.hints_provider.response = json.dumps(data)
        self.request('hints', 'filtered'); self.finish()
        saved = self.read().blocks[0].components[1].view.saved
        self.assertEqual([i.source_form for i in saved.items], ['Hello'])
        self.assertEqual((saved.assessment.accepted_count, saved.assessment.filtered_count,
                          saved.assessment.rejected_count), (1, 1, 1))
        restarted = self.make_service()
        self.assertEqual(self.read(restarted).blocks[0].components[1].view.saved, saved)
        self.assertEqual(self.request('hints', 'reuse-filtered', service=restarted).outcome, 'reuse')
        self.assertEqual(len(self.hints_provider.calls), 1)
        # A valid envelope with zero retainable candidates is saved with omissions,
        # never fabricated as a fully assessed empty result or automatically retried.
        data['candidates'] = [proposal(sid, 'absent')]
        self.hints_provider.response = json.dumps(data)
        self.request('hints', 'all-filtered', service=restarted, intent='regenerate'); self.finish(restarted)
        result = self.read().blocks[0].components[1].view.saved
        self.assertEqual(result.items, ())
        self.assertEqual(result.assessment.rejected_count, 1)
        self.assertEqual(len(self.hints_provider.calls), 2)

    def test_read_is_zero_write_zero_provider_and_no_tokenizer(self):
        before=self.snapshot()
        with patch.object(self.trans_provider,'get_model_capabilities',side_effect=AssertionError('read capacity')):
            page=self.read()
            self.assertEqual([c.view.state for c in page.blocks[0].components],['missing','missing'])
            self.assertEqual(self.service.status('book','absent').state,'missing')
            self.assertIsNone(self.service.lookup_command('book','absent'))
        self.assertEqual(self.snapshot(),before)
        self.assertFalse(self.trans_provider.calls);self.assertFalse(self.hints_provider.calls)

    def test_translation_and_hints_complete_independently_and_reuse_without_preflight(self):
        self.request();self.request('hints','hints');self.finish()
        page=self.read()
        self.assertEqual([c.view.state for c in page.blocks[0].components],['ready','ready'])
        self.assertEqual(page.blocks[0].components[1].view.saved.items,())
        with patch.object(self.translation,'prepare',side_effect=AssertionError('reuse prepare')):
            self.assertEqual(self.request(command_id='reuse').outcome,'reuse')
        self.assertEqual(len(self.trans_provider.calls),1);self.assertEqual(len(self.hints_provider.calls),1)
        self.assertEqual(self.service.status('book','generate').state,'succeeded')

    def test_duplicate_command_and_second_client_coalesce_without_duplicate_call(self):
        entered,release=self.block_provider()
        first=self.request();self.assertEqual(first.outcome,'claimed');self.assertTrue(entered.wait(5))
        other=self.make_service()
        self.assertTrue(self.request(service=other).replay)
        self.assertEqual(self.request(command_id='second',service=other).outcome,'coalesced')
        release.set();self.finish();self.finish(other)
        self.assertEqual(len(self.trans_provider.calls),1)
        self.assertEqual(other.lookup_command('book','second').operation_id,'generate')

    def test_simultaneous_clients_share_one_durable_claim(self):
        other=self.make_service();barrier=Barrier(2)
        entered,release=self.block_provider()
        def run(pair):
            service,ident=pair;barrier.wait(5)
            return self.request(command_id=ident,service=service)
        with ThreadPoolExecutor(2) as pool:
            results=list(pool.map(run,((self.service,'client-a'),(other,'client-b'))))
        self.assertEqual(sorted(r.outcome for r in results),['claimed','coalesced'])
        self.assertTrue(entered.wait(5));release.set();self.finish();self.finish(other)
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_busy_worker_still_coalesces_existing_job_and_rejects_new_work(self):
        one=self.make_service(max_workers=1);entered,release=self.block_provider()
        self.request(service=one);self.assertTrue(entered.wait(5))
        self.assertEqual(self.request(command_id='coalesce',service=one).outcome,'coalesced')
        with self.assertRaises(LearningBusy): self.request('hints','other-component',service=one)
        self.assertIsNone(self.repo.lookup_command('book','other-component'))
        release.set();self.finish(one)

    def test_detached_request_work_survives_no_wait_shutdown(self):
        entered,release=self.block_provider();submission=self.request();self.assertTrue(entered.wait(5))
        self.service.shutdown(wait=False)
        self.assertEqual(self.service.status('book',submission.operation_id).state,'running')
        release.set();self.finish()
        self.assertEqual(self.service.status('book',submission.operation_id).state,'succeeded')

    def test_provider_failure_is_unknown_and_reads_requests_do_not_retry(self):
        self.trans_provider.failure=TimeoutError('private provider body')
        self.request();self.finish()
        self.assertEqual(self.service.status('book','generate').state,'unknown')
        self.assertEqual(self.read().blocks[0].components[0].view.state,'unknown')
        self.assertEqual(self.request(command_id='again').outcome,'reconcile')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'unknown')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_explicit_unknown_retry_requires_consent_and_new_command(self):
        self.trans_provider.failure=TimeoutError();self.request();self.finish()
        self.service=self.make_service();self.trans_provider.failure=None
        self.assertEqual(self.request(command_id='no-consent',intent='retry',retry_operation_id='generate').outcome,'reconcile')
        self.request(command_id='consented',intent='retry',retry_operation_id='generate',acknowledge_possible_duplicate_cost=True)
        self.finish();self.assertEqual(len(self.trans_provider.calls),2)
        self.assertEqual(self.service.status('book','consented').state,'succeeded')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'superseded')

    def test_refresh_invalid_output_retains_previous_result_and_other_component(self):
        self.request();self.request('hints','hints');self.finish()
        before=self.read().blocks[0].components
        self.service=self.make_service();self.trans_provider.response='{}'
        self.request(command_id='refresh',intent='regenerate');self.finish()
        after=self.read().blocks[0].components
        self.assertEqual(tuple(c.view.saved for c in before),tuple(c.view.saved for c in after))
        self.assertEqual(self.service.status('book','refresh').state,'failed')
        self.assertEqual(self.service.status('book','hints').state,'succeeded')

    def test_level_change_reuses_translation_and_selects_new_hints(self):
        self.request();self.request('hints','hints');self.finish();self.settings(level='C2')
        self.assertEqual([c.view.state for c in self.read().blocks[0].components],['ready','missing'])
        self.assertEqual(self.request(command_id='after-level').outcome,'reuse')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_source_change_during_provider_fences_late_output(self):
        entered,release=self.block_provider();self.request();self.assertTrue(entered.wait(5))
        self.change();release.set();self.finish()
        self.assertEqual(self.repo.lookup_operation('book','generate').state,'stale')
        self.assertEqual(self.read().blocks[0].components[0].view.state,'missing')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_settings_change_during_provider_fences_late_output(self):
        entered,release=self.block_provider();self.request();self.assertTrue(entered.wait(5))
        self.settings(level='C2');release.set();self.finish()
        self.assertEqual(self.repo.lookup_operation('book','generate').state,'stale')
        self.assertIsNone(self.read().blocks[0].components[0].view.saved)

    def test_profile_language_change_during_provider_fences_late_output(self):
        entered,release=self.block_provider();self.request();self.assertTrue(entered.wait(5))
        self.profile.document_language_code='fr';release.set();self.finish()
        self.assertEqual(self.repo.lookup_operation('book','generate').state,'stale')
        self.assertEqual(self.read().blocks[0].components,())

    def test_settings_stale_operation_allows_explicit_consented_retry_on_current_source(self):
        entered,release=self.block_provider();self.request();self.assertTrue(entered.wait(5))
        self.settings(level='C2');release.set();self.finish()
        self.assertEqual(self.repo.lookup_operation('book','generate').state,'stale')
        self.service=self.make_service()
        self.assertEqual(self.request(command_id='ordinary').outcome,'reconcile')
        self.assertEqual(self.request(command_id='retry',intent='retry',retry_operation_id='generate',
            acknowledge_possible_duplicate_cost=True).outcome,'claimed')
        self.finish()
        self.assertEqual(self.service.status('book','retry').state,'succeeded')
        self.assertEqual(len(self.trans_provider.calls),2)

    def test_missing_or_same_language_profile_cannot_start_generation(self):
        for profile in (None,SimpleNamespace(document_language_code='zh')):
            self.profile=profile
            with self.assertRaises(LearningContractError): self.request()
        self.assertIsNone(self.repo.lookup_command('book','generate'))
        self.assertFalse(self.trans_provider.calls)

    def test_source_or_settings_expectations_are_not_authority(self):
        settings=self.preferences.read_book('book')
        self.change()
        with self.assertRaisesRegex(LearningContractError,'source_changed'):
            self.service.request('book',self.block.source.content_block_id,'translation',command_id='stale',
                expected_source=self.block.source,expected_workspace_revision=settings.workspace.revision,
                expected_book_revision=settings.book.revision)
        self.assertIsNone(self.repo.lookup_command('book','stale'))

    def test_storage_read_failure_does_not_become_missing_or_generate(self):
        with patch.object(self.repo,'read_batch',side_effect=LearningStorageError('unavailable')):
            with self.assertRaises(LearningStorageError): self.request()
        self.assertFalse(self.trans_provider.calls);self.assertIsNone(self.repo.lookup_command('book','generate'))

    def test_partial_failure_exposes_only_durable_batches(self):
        original_prepare=self.translation.prepare
        def prepare(**kw):
            p=original_prepare(**kw)
            return replace(p,plan=BatchPlan(tuple((s.sentence_id,) for s in p.sentences.sentences)))
        original_call=self.trans_provider.complete_text_with_output_budget
        def call(prompt,budget):
            if self.trans_provider.calls: self.trans_provider.failure=TimeoutError()
            return original_call(prompt,budget)
        with patch.object(self.translation,'prepare',side_effect=prepare),patch.object(self.trans_provider,'complete_text_with_output_budget',side_effect=call):
            self.request();self.finish()
        status=self.service.status('book','generate')
        self.assertEqual(status.state,'unknown');self.assertEqual(len(status.snapshot.operation.checkpoints),1)
        self.assertEqual(len(self.service.read_partial('book','generate')),1)
        self.assertIsNone(self.read().blocks[0].components[0].view.saved)

    def test_claim_ack_loss_requires_explicit_safe_reconciliation(self):
        original=self.repo.claim
        def lost(*a,**kw): original(*a,**kw);raise LearningCommitUnknown('lost')
        with patch.object(self.repo,'claim',side_effect=lost): self.assertEqual(self.request().outcome,'unknown')
        self.assertFalse(self.trans_provider.calls)
        self.assertEqual(self.service.lookup_command('book','generate').operation_id,'generate')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'resuming')
        self.finish();self.assertEqual(len(self.trans_provider.calls),1)
        self.assertEqual(self.service.status('book','generate').state,'succeeded')

    def fault_write(self,action):
        original=self.repo.write_operation;lost=[False]
        def write(doc,command,**kw):
            result=original(doc,command,**kw)
            if command.action==action and not lost[0]: lost[0]=True;raise LearningCommitUnknown('lost ack')
            return result
        return patch.object(self.repo,'write_operation',side_effect=write)

    def test_dispatch_ack_loss_never_sends_provider_even_after_reconcile(self):
        with self.fault_write('dispatch'):
            self.request();self.finish()
        self.assertFalse(self.trans_provider.calls)
        self.assertEqual(self.service.status('book','generate').state,'running')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'reconcile')
        self.now+=10001
        self.assertEqual(self.service.reconcile('book','generate').outcome,'unknown')
        self.assertFalse(self.trans_provider.calls)

    def test_checkpoint_ack_loss_recovers_full_result_without_provider_resend(self):
        with self.fault_write('checkpoint'):
            self.request();self.finish()
        self.assertEqual(self.service.status('book','generate').state,'partial')
        self.service=self.make_service();self.now+=10001
        self.assertEqual(self.service.reconcile('book','generate').outcome,'resuming')
        self.finish();self.assertEqual(self.service.status('book','generate').state,'succeeded')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_complete_checkpoint_recovery_does_not_require_provider_or_tokenizer(self):
        with self.fault_write('checkpoint'):
            self.request();self.finish()
        self.service=self.make_service()
        with patch.object(self.translation,'prepare',side_effect=AssertionError('provider unavailable')), \
                patch.object(self.trans_provider,'get_model_capabilities',side_effect=AssertionError('capacity unavailable')):
            self.assertEqual(self.service.reconcile('book','generate').outcome,'resuming')
            self.finish()
        self.assertEqual(self.service.status('book','generate').state,'succeeded')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_hints_complete_checkpoint_recovery_is_independent_and_offline(self):
        with self.fault_write('checkpoint'):
            self.request('hints','hints');self.finish()
        self.service=self.make_service()
        with patch.object(self.hints,'prepare',side_effect=AssertionError('provider unavailable')):
            self.assertEqual(self.service.reconcile('book','hints').outcome,'resuming')
            self.finish()
        self.assertEqual(self.service.status('book','hints').state,'succeeded')
        self.assertEqual(self.read().blocks[0].components[1].view.saved.items,())
        self.assertFalse(self.trans_provider.calls)
        self.assertEqual(len(self.hints_provider.calls),1)

    def test_commit_ack_loss_status_confirms_success_without_resend(self):
        with self.fault_write('commit'):
            self.request();self.finish()
        self.assertEqual(self.service.status('book','generate').state,'succeeded')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'succeeded')
        self.assertTrue(self.request().replay)
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_superseded_command_cannot_be_reinterpreted_as_other_component(self):
        self.request();self.finish()
        with self.assertRaisesRegex(LearningContractError,'command_conflict'): self.request('hints')
        self.assertFalse(self.hints_provider.calls)

    def test_policy_denies_initial_claim_with_no_model(self):
        policy=MutablePolicy();policy.allowed=False
        repo=self.policy_repository(policy);service=self.make_service(repo)
        result=self.request(service=service)
        self.assertEqual(result.outcome,'denied');self.assertEqual(result.reason,'test_restricted')
        self.assertFalse(self.trans_provider.calls);self.assertEqual(policy.calls,1)

    def test_policy_is_rechecked_before_each_paid_batch(self):
        policy=MutablePolicy();repo=self.policy_repository(policy);service=self.make_service(repo)
        prepare=self.translation.prepare;call=self.trans_provider.complete_text_with_output_budget
        def split(**kw):
            p=prepare(**kw);return replace(p,plan=BatchPlan(tuple((s.sentence_id,) for s in p.sentences.sentences)))
        def restrict(prompt,budget):
            result=call(prompt,budget);policy.allowed=False;return result
        with patch.object(self.translation,'prepare',side_effect=split),patch.object(self.trans_provider,'complete_text_with_output_budget',side_effect=restrict):
            self.request(service=service);self.finish(service)
        self.assertEqual(len(self.trans_provider.calls),1);self.assertEqual(policy.calls,3)
        self.assertEqual(service.status('book','generate').state,'failed')
        self.assertEqual(len(service.read_partial('book','generate')),1)

    def test_page_limits_and_foreign_blocks_fail_without_write(self):
        before=self.snapshot()
        for ids in (('unknown',),(self.block.source.content_block_id,)*2,tuple(str(i) for i in range(51))):
            with self.assertRaises(LearningContractError): self.service.read('book',ids)
        self.assertEqual(self.snapshot(),before)


    def test_normal_mode_has_no_learning_projection_or_generation(self):
        self.settings(normal=True)
        before=self.snapshot()
        self.assertEqual(self.read().blocks[0].components,())
        with self.assertRaisesRegex(LearningContractError,'mode_unavailable'): self.request()
        self.assertEqual(self.snapshot(),before);self.assertFalse(self.trans_provider.calls)

    def test_profile_change_after_preflight_is_rejected_by_locked_claim_validator(self):
        prepare=self.translation.prepare
        def change(**kw):
            result=prepare(**kw);self.profile.document_language_code='fr';return result
        with patch.object(self.translation,'prepare',side_effect=change):
            with self.assertRaises(LearningContractError): self.request()
        self.assertIsNone(self.repo.lookup_command('book','generate'))
        self.assertFalse(self.trans_provider.calls)

    def test_profile_change_after_claim_prevents_dispatch(self):
        start=self.service.worker.start
        def change(*args):
            self.profile.document_language_code='fr';return start(*args)
        with patch.object(self.service.worker,'start',side_effect=change):
            self.request();self.finish()
        op=self.repo.lookup_operation('book','generate')
        self.assertEqual(op.state,'stale');self.assertFalse(op.outcome_uncertain)
        self.assertFalse(self.trans_provider.calls)

    def test_executor_rejection_records_known_failure_without_call(self):
        with patch.object(self.service.worker,'start',side_effect=RuntimeError('executor closed')):
            with self.assertRaises(RuntimeError): self.request()
        self.assertEqual(self.service.status('book','generate').state,'failed')
        self.assertFalse(self.trans_provider.calls)
        self.assertTrue(self.request().replay)

    def test_claim_expiry_is_explicit_and_does_not_generate(self):
        claim=self.repo.claim
        def lost(*args,**kw): claim(*args,**kw);raise LearningCommitUnknown('lost')
        with patch.object(self.repo,'claim',side_effect=lost): self.request()
        self.now+=10001
        before=self.snapshot()
        self.assertEqual(self.service.status('book','generate').state,'claimed')
        self.assertEqual(self.snapshot(),before)
        self.assertEqual(self.service.reconcile('book','generate').outcome,'failed')
        self.assertFalse(self.trans_provider.calls)

    def test_simultaneous_reconciliation_dispatches_only_once(self):
        claim=self.repo.claim
        def lost(*args,**kw): claim(*args,**kw);raise LearningCommitUnknown('lost')
        with patch.object(self.repo,'claim',side_effect=lost): self.request()
        other=self.make_service();barrier=Barrier(2);entered,release=self.block_provider()
        def resume(service):
            barrier.wait(5);return service.reconcile('book','generate')
        with ThreadPoolExecutor(2) as pool:
            results=list(pool.map(resume,(self.service,other)))
        self.assertTrue(all(r.outcome in ('resuming','reconcile') for r in results))
        self.assertTrue(entered.wait(5));release.set();self.finish();self.finish(other)
        self.assertEqual(len(self.trans_provider.calls),1)
        self.assertEqual(self.service.status('book','generate').state,'succeeded')

    def test_partial_checkpoint_recovery_calls_only_the_remaining_batch(self):
        prepare=self.translation.prepare
        def split(**kw):
            p=prepare(**kw);return replace(p,plan=BatchPlan(tuple((s.sentence_id,) for s in p.sentences.sentences)))
        with patch.object(self.translation,'prepare',side_effect=split):
            with self.fault_write('checkpoint'):
                self.request();self.finish()
            self.assertEqual(len(self.trans_provider.calls),1)
            self.assertEqual(len(self.service.read_partial('book','generate')),1)
            self.service=self.make_service()
            self.service.reconcile('book','generate');self.finish()
        self.assertEqual(len(self.trans_provider.calls),2)
        self.assertEqual(self.service.status('book','generate').state,'succeeded')

    def test_settings_change_after_unknown_preserves_uncertain_cost_on_reconcile(self):
        self.trans_provider.failure=TimeoutError();self.request();self.finish()
        self.settings(level='C2')
        self.assertEqual(self.service.reconcile('book','generate').outcome,'stale')
        op=self.repo.lookup_operation('book','generate')
        self.assertTrue(op.outcome_uncertain);self.assertEqual(op.state,'stale')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_policy_reuse_does_not_consume_new_admission(self):
        policy=MutablePolicy();repo=self.policy_repository(policy);service=self.make_service(repo)
        self.request(service=service);self.finish(service);count=policy.calls
        policy.allowed=False
        self.assertEqual(self.request(command_id='reuse-restricted',service=service).outcome,'reuse')
        self.assertEqual(policy.calls,count);self.assertEqual(len(self.trans_provider.calls),1)

    def test_storage_failure_after_dispatch_does_not_fabricate_success(self):
        write=self.repo.write_operation
        def fail(doc,command,**kw):
            if command.action=='checkpoint': raise LearningStorageError('known storage failure')
            return write(doc,command,**kw)
        with patch.object(self.repo,'write_operation',side_effect=fail):
            self.request();self.finish()
        self.assertEqual(self.service.status('book','generate').state,'running')
        self.assertIsNone(self.read().blocks[0].components[0].view.saved)
        self.assertEqual(self.service.read_partial('book','generate'),())
        self.assertEqual(self.service.reconcile('book','generate').outcome,'reconcile')
        self.assertEqual(len(self.trans_provider.calls),1)

    def test_settings_race_during_page_read_is_conflict_not_missing(self):
        read=self.repo.read_batch
        def change(*args,**kw):
            result=read(*args,**kw);self.settings(level='C2');return result
        with patch.object(self.repo,'read_batch',side_effect=change):
            with self.assertRaisesRegex(LearningContractError,'settings_changed'): self.read()
        self.assertFalse(self.trans_provider.calls)

    def test_worker_logs_do_not_expose_private_book_or_provider_error(self):
        self.trans_provider.failure=TimeoutError('SECRET_PROVIDER_ERROR_BODY')
        with self.assertLogs('learning.worker',level='INFO') as logs:
            self.request();self.finish()
        text='\n'.join(logs.output)
        self.assertIn('outcome=unknown',text)
        self.assertNotIn('SECRET_PROVIDER_ERROR_BODY',text)
        self.assertNotIn(self.block.effective_text,text)


class FileLearningOrchestrationTests(OrchestrationCases,unittest.TestCase):
    def setUp(self):
        files.LearningFileRepositoryTests.setUp(self)
        self.lifecycle=FileLearningDocumentLifecycle(self.docs,self.repo,self.prefs)
        self.setup_service()
    def snapshot(self): return {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
    def policy_repository(self,policy):
        from document_structure.learning_file_repository import FileLearningRepository
        return FileLearningRepository(self.root/'learning',self.docs,self.prefs,admission=policy)


class SqlLearningOrchestrationTests(OrchestrationCases,unittest.TestCase):
    setUpClass=classmethod(sql.LearningPostgresTests.setUpClass.__func__)
    drop_schema=classmethod(sql.LearningPostgresTests.drop_schema.__func__)
    def setUp(self):
        sql.LearningPostgresTests.setUp(self)
        self.setup_service()
    snapshot=sql.LearningPostgresTests.snapshot
    def test_final_sql_write_failure_rolls_back_output_and_keeps_previous_result(self):
        from db import learning_rows
        self.request();self.finish()
        before=self.read().blocks[0].components[0].view.saved
        self.service=self.make_service();persist=learning_rows.persist
        def fail(connection,row_id,scope,doc,state,previous):
            persist(connection,row_id,scope,doc,state,previous)
            if any(op.operation_id=='refresh' and op.state=='succeeded' for op in state['operations'].values()):
                raise RuntimeError('injected after SQL writes')
        with patch.object(learning_rows,'persist',side_effect=fail):
            self.request(command_id='refresh',intent='regenerate');self.finish()
        self.assertEqual(self.read().blocks[0].components[0].view.saved,before)
        self.assertEqual(self.service.status('book','refresh').state,'failed')
        self.assertEqual(len(self.trans_provider.calls),2)

    def policy_repository(self,policy):
        from db.postgres_learning_repository import PostgresLearningRepository
        return PostgresLearningRepository(self.scoped,self.repo.scope,admission=policy)


if __name__=='__main__': unittest.main()
