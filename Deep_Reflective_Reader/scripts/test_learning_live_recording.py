"""LLRN-22 recording/replay must preserve production failure classification."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dependency_injector import providers
import httpx
from openai import OpenAI

from llm.openai_llm_provider import OpenAILLMProvider
from config.learning_runtime import LearningProvider
from scripts.accept_learning_live import RecordingProvider, ReplayProvider
from scripts.test_learning_container_wiring import OnceProvider
from scripts import test_learning_routes as routes
from scripts import test_learning_hints_service as hints
from scripts import test_learning_translation_service as translation


def report(raw=None, **call):
    if raw is not None:
        call['response_text'] = raw
    return {'model': 'offline-test', 'provider_calls': [call]}


def prompt(sid='new'):
    return 'INPUT_JSON:\n'+json.dumps({'sentences': [{'sentence_id': sid}]})


class RecordingTests(unittest.TestCase):
    def test_valid_text_is_forwarded_verbatim_with_optional_decoded_view(self):
        source = OnceProvider()
        source.response = ' \n{"text": "繁體中文", "n": 1}  '
        recorder = RecordingProvider(source)
        self.assertEqual(recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens), source.response)
        self.assertEqual(recorder.calls[0]['response_text'], source.response)
        self.assertEqual(recorder.calls[0]['response'], json.loads(source.response))
        self.assertEqual(recorder.calls[0]['status'], 'returned')
        self.assertEqual(len(source.calls), 1)

    def test_invalid_or_deep_json_still_returns_raw_text(self):
        for raw in ('', '{bad', '```json\n{}\n```', '['*2000+'0'+']'*2000):
            with self.subTest(raw=raw[:20]):
                source = OnceProvider(); source.response = raw
                recorder = RecordingProvider(source)
                self.assertEqual(recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens), raw)
                record = recorder.calls[0]
                self.assertEqual(record['status'], 'returned')
                self.assertEqual(record['response_text'], raw)
                self.assertIn(record['response_decode_error'], ('JSONDecodeError', 'RecursionError'))
                self.assertNotIn('response', record)
                self.assertNotIn('error_type', record)

    def test_provider_exception_is_same_exception_without_retry_or_private_message(self):
        source = OnceProvider(); source.failure = TimeoutError('private connection details')
        recorder = RecordingProvider(source)
        with self.assertRaises(TimeoutError) as caught:
            recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens)
        self.assertIs(caught.exception, source.failure)
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(recorder.calls[0]['status'], 'failed_or_unknown')
        self.assertNotIn('response_text', recorder.calls[0])
        self.assertNotIn('private connection details', json.dumps(recorder.calls))

    def test_spending_bound_counts_returned_invalid_payloads_without_ninth_call(self):
        source = OnceProvider(); source.response = '{bad'
        recorder = RecordingProvider(source)
        for _ in range(8):
            recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens)
        with self.assertRaisesRegex(RuntimeError, 'acceptance_call_bound_exceeded'):
            recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens)
        self.assertEqual(len(source.calls), 8)
        self.assertEqual(len(recorder.calls), 8)

    def test_actual_sdk_receipts_survive_invalid_json_on_both_endpoints(self):
        for model in ('gpt-4.1-mini', 'gpt-4o-mini'):
            with self.subTest(model=model):
                source = OpenAILLMProvider(SimpleNamespace(get=lambda: 'offline-test-key'), model=model)
                recorder = RecordingProvider(source)
                requests, clients = [], []
                def handler(request):
                    requests.append(request)
                    if model == 'gpt-4.1-mini':
                        payload = {'id': 'receipt', 'object': 'response', 'created_at': 0, 'model': model,
                            'status': 'completed', 'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
                            'output': [{'type': 'message', 'id': 'message', 'role': 'assistant',
                                'status': 'completed', 'content': [{'type': 'output_text', 'text': '{bad', 'annotations': []}]}]}
                    else:
                        payload = {'id': 'receipt', 'object': 'chat.completion', 'created': 0, 'model': model,
                            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
                            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{bad'}}]}
                    return httpx.Response(200, json=payload, headers={'x-request-id': 'request-receipt'})
                def factory(**kwargs):
                    self.assertEqual(kwargs['max_retries'], 0)
                    client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                    clients.append(client)
                    return client
                with patch('llm.openai_llm_provider.OpenAIClient', side_effect=factory):
                    self.assertEqual(recorder.complete_text_once_with_output_budget(prompt(), source.get_model_capabilities().max_output_tokens), '{bad')
                record = recorder.calls[0]
                self.assertEqual(record['request_id'], 'request-receipt')
                self.assertEqual(record['response_id'], 'receipt')
                self.assertEqual(record['model'], model)
                self.assertEqual(record['usage']['total_tokens'], 5)
                self.assertEqual(record['status'], 'returned')
                self.assertEqual(len(requests), 1)
                self.assertTrue(clients[0].is_closed())


class ReplayTests(unittest.TestCase):
    def test_malformed_or_wrong_shape_is_preserved_for_production_validation(self):
        for raw in ('', '{bad', '```json\n{}\n```', 'null', '[]', '{"rows": null}', '{"rows":[4]}'):
            with self.subTest(raw=raw):
                replay = ReplayProvider(report(raw))
                self.assertEqual(replay.complete_text_once_with_output_budget(prompt(), 32768), raw)
                self.assertEqual(replay.calls[0]['response_source'], 'raw_text')
                self.assertEqual(replay.calls[0]['status'], 'returned')

    def test_remap_preserves_duplicate_keys_whitespace_and_unrelated_values(self):
        raw = ('{ "status":"bad", "status":"completed",\n'
               '"rows":[{"sentence_id":"old", "translated_text":"old"}], '
               '"candidates":[{"sentence\\u005fid":"old", "meaning":"old"}] }')
        replay = ReplayProvider(report(raw))
        actual = replay.complete_text_once_with_output_budget(prompt(), 32768)
        expected = raw.replace('"sentence_id":"old"', '"sentence_id":"new"').replace(
            '"sentence\\u005fid":"old"', '"sentence\\u005fid":"new"')
        self.assertEqual(actual, expected)
        self.assertEqual(actual.count('"status"'), 2)
        self.assertEqual(replay.calls[0]['response_text'], expected)

    def test_raw_takes_precedence_over_legacy_decoded_response(self):
        replay = ReplayProvider(report('{bad', response={'rows': [{'sentence_id': 'old'}]}))
        self.assertEqual(replay.complete_text_once_with_output_budget(prompt(), 32768), '{bad')

    def test_legacy_report_is_remapped_and_marked_lossy(self):
        replay = ReplayProvider(report(response={'rows': [{'sentence_id': 'old', 'translated_text': '中文'}]},
                                       request_id='historical'))
        value = json.loads(replay.complete_text_once_with_output_budget(prompt(), 32768))
        self.assertEqual(value['rows'][0]['sentence_id'], 'new')
        self.assertEqual(replay.calls[0]['response_source'], 'legacy_json')
        self.assertEqual(replay.calls[0]['original_request_id'], 'historical')
        chained = ReplayProvider({'model': 'offline-test', 'provider_calls': replay.calls})
        chained.complete_text_once_with_output_budget(prompt('next'), 32768)
        self.assertEqual(chained.calls[0]['original_request_id'], 'historical')
        self.assertEqual(chained.calls[0]['response_source'], 'raw_text')

    def test_missing_historical_output_is_not_fabricated(self):
        replay = ReplayProvider(report(status='failed_or_unknown'))
        with self.assertRaisesRegex(ValueError, 'acceptance_replay_response_missing'):
            replay.complete_text_once_with_output_budget(prompt(), 32768)
        self.assertEqual(replay.calls, [])

    def test_recorded_duplicate_keys_remain_known_invalid_in_both_services(self):
        for module, error in ((translation, translation.LearningTranslationError), (hints, hints.LearningHintsError)):
            with self.subTest(module=module.__name__):
                source = OnceProvider()
                recorder = RecordingProvider(source)
                service, prepared = module.prepare(LearningProvider(lambda: recorder))
                service.generate_batch(prepared, translation.operation(prepared), now_ms=2)
                raw = recorder.calls[0]['response_text'].replace('{', '{"status":"completed",', 1)
                # Sentence IDs change with a different source; raw replay remaps them.
                replay = ReplayProvider(report(raw))
                service, prepared = module.prepare(LearningProvider(lambda: replay), text='Different! Paragraph.')
                with self.assertRaises(error) as caught:
                    service.generate_batch(prepared, translation.operation(prepared), now_ms=2)
                self.assertTrue(caught.exception.outcome_known)
                self.assertIn('invalid_output', caught.exception.code)
                self.assertEqual(len(replay.calls), 1)


class RecordedHttpCases:
    prepare = routes.HttpCases.prepare
    post = routes.HttpCases.post
    payload = routes.HttpCases.payload
    read = routes.HttpCases.read
    finish = routes.HttpCases.finish

    def observe(self):
        recorder = RecordingProvider(self.provider)
        self.container.llm_provider.override(providers.Object(recorder))
        return recorder

    def test_invalid_refresh_is_known_preserves_saved_and_replays_without_calls(self):
        recorder = self.observe()
        for index, component in enumerate(('translation', 'hints')):
            self.provider.response = None
            initial = self.payload(component+'-initial') | {'component': component}
            self.post(routes.BASE+'generate', initial)
            self.assertEqual(self.finish(initial['command_id']), 'succeeded')
            saved = self.read()['blocks'][0]['components'][index]['saved']
            self.provider.response = '{bad'
            body = self.payload(component+'-refresh', intent='regenerate') | {'component': component}
            receipt = self.post(routes.BASE+'generate', body)
            self.assertEqual(self.finish(body['command_id']), 'failed')
            count = len(recorder.calls)
            status = self.post(routes.BASE+'operations/read', {'handle': receipt['operation']})
            self.assertEqual(status['failure_code'], 'invalid_output')
            self.assertFalse(status['outcome_uncertain'])
            self.assertEqual(self.read()['blocks'][0]['components'][index]['saved'], saved)
            self.assertEqual(self.post(routes.BASE+'generate', body), receipt | {'replay': True})
            self.assertEqual(len(recorder.calls), count)
            self.assertEqual(recorder.calls[-1]['response_text'], '{bad')
            self.assertEqual(recorder.calls[-1]['status'], 'returned')
            # A known failure can be explicitly retried without unknown-cost consent.
            self.provider.response = None
            retry = self.payload(component+'-retry', intent='retry', retry_operation=receipt['operation']) | {'component': component}
            self.assertEqual(self.post(routes.BASE+'generate', retry)['outcome'], 'claimed')
            self.assertEqual(self.finish(retry['command_id']), 'succeeded')
            self.assertEqual(len(recorder.calls), count+1)

    def test_actual_provider_exception_stays_unknown_and_never_auto_retries(self):
        recorder = self.observe()
        self.provider.failure = TimeoutError('private upstream failure')
        receipt = self.post(routes.BASE+'generate', self.payload())
        self.assertEqual(self.finish(), 'unknown')
        status = self.post(routes.BASE+'operations/read', {'handle': receipt['operation']})
        self.assertEqual(status['failure_code'], 'provider_outcome_unknown')
        self.assertTrue(status['outcome_uncertain'])
        self.assertIsNone(self.read()['blocks'][0]['components'][0]['saved'])
        retry = self.payload('retry', intent='retry', retry_operation=receipt['operation'])
        self.assertEqual(self.post(routes.BASE+'generate', retry)['outcome'], 'reconcile')
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertNotIn('private upstream failure', json.dumps(status))


class FileRecordedHttpTests(RecordedHttpCases, unittest.TestCase):
    setUp = routes.FileHttpTests.setUp


class PostgresRecordedHttpTests(RecordedHttpCases, unittest.TestCase):
    setUpClass = classmethod(routes.PostgresHttpTests.setUpClass.__func__)
    drop_schema = classmethod(routes.PostgresHttpTests.drop_schema.__func__)
    setUp = routes.PostgresHttpTests.setUp


if __name__ == '__main__':
    unittest.main()
