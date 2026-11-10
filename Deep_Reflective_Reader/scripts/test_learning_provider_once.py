"""Real SDK with HTTP mock transport: durable dispatch cannot hide retries."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI, APIStatusError, APITimeoutError

from llm.openai_llm_provider import OpenAILLMProvider


class OnceProviderTests(unittest.TestCase):
    def provider(self, model):
        return OpenAILLMProvider(SimpleNamespace(get=lambda: 'offline-test-key'), model=model)

    def run_call(self, provider, handler):
        clients, options = [], []
        def factory(**kwargs):
            options.append(kwargs)
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client
        try:
            with patch('llm.openai_llm_provider.OpenAIClient', side_effect=factory):
                return provider.complete_text_once_with_output_budget('test prompt', provider.model_capabilities.max_output_tokens)
        finally:
            self.assertEqual(len(clients), 1)
            self.assertTrue(clients[0].is_closed())
            self.assertEqual(options[0]['max_retries'], 0)
            self.assertEqual(options[0]['timeout'], 300.0)
            self.assertEqual(options[0]['base_url'], provider.llm.api_base)

    def test_both_endpoint_families_use_existing_model_capacity_and_single_request(self):
        for model in ('gpt-4.1-mini', 'gpt-4o-mini'):
            with self.subTest(model=model):
                provider = self.provider(model)
                retries = provider.llm.max_retries
                budget = provider.effective_output_budget
                calls = []
                def handler(request):
                    calls.append(request)
                    body = json.loads(request.content)
                    self.assertEqual(body['model'], model)
                    if model == 'gpt-4.1-mini':
                        self.assertEqual(request.url.path, '/v1/responses')
                        self.assertEqual(body['input'], 'test prompt')
                        self.assertEqual(body['max_output_tokens'], provider.model_capabilities.max_output_tokens)
                        payload = {'id': 'test', 'object': 'response', 'created_at': 0, 'model': model,
                            'status': 'completed', 'output': [{'type': 'message', 'id': 'message', 'role': 'assistant',
                                'status': 'completed', 'content': [{'type': 'output_text', 'text': ' done ', 'annotations': []}]}]}
                    else:
                        self.assertEqual(request.url.path, '/v1/chat/completions')
                        self.assertEqual(body['messages'], [{'role': 'user', 'content': 'test prompt'}])
                        self.assertEqual(body['max_completion_tokens'], provider.model_capabilities.max_output_tokens)
                        payload = {'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': model,
                            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': ' done '}}]}
                    return httpx.Response(200, json=payload)
                self.assertEqual(self.run_call(provider, handler), 'done')
                self.assertEqual(len(calls), 1)
                self.assertEqual(provider.llm.max_retries, retries)
                self.assertEqual(provider.effective_output_budget, budget)

    def test_retryable_errors_are_never_retried(self):
        for model in ('gpt-4.1-mini', 'gpt-4o-mini'):
            for status in (429, 500, 503):
                with self.subTest(model=model, status=status):
                    calls = []
                    def handler(request):
                        calls.append(request)
                        return httpx.Response(status, json={'error': {'message': 'test', 'type': 'test', 'code': 'test'}})
                    with self.assertRaises(APIStatusError):
                        self.run_call(self.provider(model), handler)
                    self.assertEqual(len(calls), 1)

    def test_timeout_is_one_unknown_attempt(self):
        for model in ('gpt-4.1-mini', 'gpt-4o-mini'):
            with self.subTest(model=model):
                calls = []
                def handler(request):
                    calls.append(request)
                    raise httpx.ReadTimeout('test', request=request)
                with self.assertRaises(APITimeoutError):
                    self.run_call(self.provider(model), handler)
                self.assertEqual(len(calls), 1)

    def test_invalid_budget_never_opens_client(self):
        provider = self.provider('gpt-4.1-mini')
        with patch('llm.openai_llm_provider.OpenAIClient') as client:
            for budget in (0, True, provider.model_capabilities.max_output_tokens+1):
                with self.assertRaisesRegex(ValueError, 'invalid_output_budget'):
                    provider.complete_text_once_with_output_budget('test', budget)
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
