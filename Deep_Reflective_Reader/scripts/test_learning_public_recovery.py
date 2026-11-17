"""LEARN-03: production HTTP projections and explicit retry over real file/SQL."""
from types import SimpleNamespace
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scripts.test_learning_recovery_provenance import RecoveryCases, FileRecoveryTests, PostgresRecoveryTests
from learning_routes import router
from api_learning_schemas import LearningOperationResponse, LearningSubmissionResponse, learning_source_token


class PublicRecoveryCases:
    setup_service = RecoveryCases.setup_service
    make_service = RecoveryCases.make_service
    request = RecoveryCases.request
    finish = RecoveryCases.finish
    read = RecoveryCases.read
    fault_write = RecoveryCases.fault_write
    settings = RecoveryCases.settings
    change = RecoveryCases.change
    prepare_partial = RecoveryCases.prepare_partial
    restart = RecoveryCases.restart
    saved = RecoveryCases.saved
    complete_partial = RecoveryCases.complete_partial

    def successful(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def public_failure(self, code):
        for component in ('translation', 'hints'):
            with self.subTest(component=component, code=code):
                if code == 'recovery_incompatible':
                    self.prepare_partial(component, previous=True)
                    provider = self.restart(component, model='model-B')
                else:
                    provider, _ = self.complete_partial(component, mixed=True)
                    self.now += 10001
                saved = self.saved(component)
                parts = self.repo.read_checkpoints('book', self.op)
                receipt = self.repo.lookup_command('book', self.op)
                calls = len(provider.calls)
                fence = self.repo.inspect_operation('book', self.op).operation.fence
                app = FastAPI(); app.include_router(router)
                app.state.learning_runtime = SimpleNamespace(service=self.service)
                prefix = '/documents/reading-interactions/learning/'
                handle = {'doc_name': 'book', 'operation_id': self.op}
                with TestClient(app) as client:
                    submission = self.successful(client.post(prefix+'operations/reconcile', json={'handle': handle}))
                    self.assertEqual(submission['outcome'], 'failed' if code == 'recovery_incompatible' else 'resuming')
                    self.assertIsNone(submission['reason'])
                    self.finish()
                    for endpoint in ('read', 'wait'):
                        status = self.successful(client.post(prefix+'operations/'+endpoint, json={'handle': handle}))
                        self.assertEqual((status['state'], status['failure_code']), ('failed', code))
                        self.assertFalse(status['outcome_uncertain'])
                        self.assertTrue(status['source_current'] and status['settings_current'] and status['owns_current_slot'])
                    for invalid in ('future_failure', 1, {}, True):
                        with self.assertRaises(ValidationError):
                            LearningOperationResponse.model_validate({**status, 'failure_code': invalid})
                    with self.assertRaises(ValidationError):
                        LearningSubmissionResponse.model_validate({**submission, 'reason': code})
                    partial = self.successful(client.post(prefix+'operations/partial', json={'handle': handle}))
                    self.assertEqual((partial['completed_batches'], len(partial['batches'])), (len(parts), len(parts)))
                    lookup = self.successful(client.post(prefix+'commands/read', json={'doc_name': 'book', 'command_id': self.op}))
                    self.assertTrue(lookup['found'])
                    self.assertEqual(lookup['receipt']['outcome'], 'claimed')
                    again = self.successful(client.post(prefix+'operations/reconcile', json={'handle': handle}))
                    self.assertEqual(again['outcome'], 'failed')
                    self.assertEqual(len(provider.calls), calls)
                    self.assertEqual(self.saved(component), saved)
                    self.assertEqual(self.repo.read_checkpoints('book', self.op), parts)
                    self.assertEqual(self.repo.lookup_command('book', self.op), receipt)
                    self.service = self.make_service(); app.state.learning_runtime.service = self.service
                    current = self.preferences.read_book('book')
                    source = self.service.source_binding('book', self.block.source.content_block_id)
                    command = dict(doc_name='book', content_block_id=source.content_block_id, component=component,
                        command_id=self.op+'-retry', intent='retry', retry_operation=handle,
                        expected_source_token=learning_source_token(source),
                        expected_workspace_revision=current.workspace.revision, expected_book_revision=current.book.revision)
                    retry = self.successful(client.post(prefix+'generate', json=command))
                    self.assertEqual(retry['outcome'], 'claimed')
                    self.finish()
                    result = self.successful(client.post(prefix+'operations/read', json={'handle': retry['operation']}))
                    self.assertEqual(result['state'], 'succeeded')
                    self.assertEqual(len(provider.calls), calls+2)
                    self.assertGreater(self.repo.inspect_operation('book', command['command_id']).operation.fence, fence)
                    self.assertEqual(self.saved(component).context.provenance.model, 'model-B')
                    self.assertEqual(self.repo.read_checkpoints('book', self.op), parts)
                    self.assertEqual(self.repo.lookup_command('book', self.op), receipt)
                    reloaded = self.successful(client.post(prefix+'read', json={'doc_name': 'book', 'content_block_ids': [source.content_block_id]}))
                    view = next(v for v in reloaded['blocks'][0]['components'] if v['component'] == component)
                    self.assertEqual(view['state'], 'ready')
                    self.assertEqual(view['saved']['result_revision'], saved.context.result_revision+1)
                    self.assertEqual(len(provider.calls), calls+2)

    def test_incompatible_recovery_public_contract(self):
        self.public_failure('recovery_incompatible')

    def test_invalid_assembly_public_contract(self):
        self.public_failure('assembly_invalid')


class FilePublicRecoveryTests(PublicRecoveryCases, unittest.TestCase):
    setUp = FileRecoveryTests.setUp


class PostgresPublicRecoveryTests(PublicRecoveryCases, unittest.TestCase):
    setUpClass = classmethod(PostgresRecoveryTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresRecoveryTests.drop_schema.__func__)
    setUp = PostgresRecoveryTests.setUp


if __name__ == '__main__':
    unittest.main()
