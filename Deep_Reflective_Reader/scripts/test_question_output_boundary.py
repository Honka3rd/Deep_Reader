"""QUESTION-03/04: completed-output admission and non-content diagnostics.

Actual question/context/QA code and the unchanged ask route; provider and runtime
bundle ports are deterministic. No main lifespan, storage, network or paid model.
"""
import ast
import contextlib
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from api_schemas import AskDocumentRequest, AskDocumentResponse
from app.qa_coordinator import QACoordinator
from context.context_orchestrator import ContextOrchestrator
from embeddings.embedding_similarity_service import EmbeddingSimilarityService
from language.language_code import LanguageCode
from question.qa_enums import QuestionScope
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.question_scope_resolver import QuestionScopeResolver
from question.standardized.question_standardizer import QuestionStandardizer
from question.standardized.standardized_question import StandardizedQuestion
from test_question_scope_routing import AnswerMode, AnswerLevel, Budget, Builder, Bundle, Coverage, Embedder

MARKER = 'SYNTHETIC_PRIVATE_QUESTION_OUTPUT_03_04'
BACKEND = Path(__file__).resolve().parents[1]


class Provider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete_text(self, prompt):
        self.calls.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def pipeline(reply):
    provider = Provider(reply)
    bundle = Bundle()
    relevance = Mock()
    scope = Mock()
    context = ContextOrchestrator(QuestionStandardizer(provider), relevance,
        scope, Coverage(), Budget(), Builder())
    coordinator = object.__new__(QACoordinator)
    coordinator.document_preparation_pipeline = SimpleNamespace(
        prepare_and_load=Mock(return_value=SimpleNamespace(bundle=bundle,
            assets=SimpleNamespace(structured_document_path=None, bundle_ready=True,
                                   faiss_ready=True, profile_ready=True))))
    coordinator.context_orchestrator = context
    coordinator.prompt_assembler = Mock()
    coordinator.llm_provider = Provider('answer')
    coordinator.session_manager = SimpleNamespace(get_or_create_session=Mock(
        return_value=SimpleNamespace(active_chunk_index=None)))
    return coordinator, provider, bundle, relevance, scope


def route_app(coordinator):
    """Compile exactly the production function/decorator, with real schemas."""
    module = ast.parse((BACKEND / 'main.py').read_text())
    route = next(node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'ask_document')
    app = FastAPI()
    namespace = dict(app=app, Response=Response, HTTPException=HTTPException,
        AskDocumentRequest=AskDocumentRequest, AskDocumentResponse=AskDocumentResponse,
        uuid4=uuid4, qa_coordinator=coordinator)
    exec(compile(ast.Module(body=[route], type_ignores=[]), str(BACKEND / 'main.py'), 'exec'), namespace)
    return app


class QuestionOutputBoundaryTests(unittest.TestCase):
    def reject(self, reply, exception=RuntimeError):
        provider = Provider(reply)
        with self.assertRaises(Exception) as caught:
            QuestionStandardizer(provider).standardize(MARKER, 'en')
        self.assertIsInstance(caught.exception, exception)
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn(MARKER, str(caught.exception))
        self.assertTrue(str(caught.exception).startswith('question_standardization_'))
        return caught.exception

    def test_root_shapes_and_nonstring_provider_results(self):
        for reply in ('[]', 'null', 'true', '42', '"'+MARKER+'"', None, {}, [], 12, True):
            with self.subTest(reply=reply):
                self.reject(reply)

    def test_missing_blank_and_wrong_type_required_fields(self):
        for field in ('user_language', 'standardized_query'):
            for value in (None, False, True, 1, 1.5, [], ['en'], {}, {'en': 1}, '', ' \t\n\u2003'):
                with self.subTest(field=field, value=value):
                    payload = dict(user_language='en', standardized_query='Explain this sentence.')
                    payload[field] = value
                    self.reject(json.dumps(payload))
            payload = dict(user_language='en', standardized_query='Question')
            del payload[field]
            self.reject(json.dumps(payload))
        self.reject('{}')

    def test_duplicate_keys_including_nested_and_escaped_names(self):
        for reply in (
            '{"user_language":"en","user_language":"zh","standardized_query":"Q"}',
            '{"user_language":"en","standardized_query":"Q","standardized_query":"'+MARKER+'"}',
            '{"user_language":"en","\\u0075ser_language":"zh","standardized_query":"Q"}',
            '{"user_language":"en","standardized_query":"Q","extra":{"x":1,"x":2}}',
        ):
            with self.subTest(reply=reply):
                self.reject(reply)

    def test_nonfinite_values_rejected_anywhere_including_exponent_overflow(self):
        for token in ('NaN', 'Infinity', '-Infinity', '1e999', '-1e999'):
            for tail in ('"standardized_query":'+token,
                         '"standardized_query":"Q","extra":[{"value":'+token+'}]'):
                with self.subTest(token=token, tail=tail):
                    self.reject('{"user_language":"en",'+tail+'}')

    def test_invalid_json_and_nested_decoder_failure_are_safe(self):
        for reply in (MARKER, '{"'+MARKER+'":}', '{', '[' * 2000 + '0' + ']' * 2000):
            with self.subTest(size=len(reply)):
                self.reject(reply)

    def test_unknown_language_is_explicit_and_not_fabricated(self):
        for language in ('unknown', 'cannot determine', '.', MARKER):
            with self.subTest(language=language):
                self.reject(json.dumps(dict(user_language=language, standardized_query='Question')), ValueError)

    def test_invalid_document_language_does_not_dispatch_or_echo(self):
        provider = Provider('unused')
        with self.assertRaisesRegex(ValueError, '^question_standardization_unknown_document_language$'):
            QuestionStandardizer(provider).standardize('Question', MARKER)
        self.assertEqual(provider.calls, [])

    def test_valid_codes_aliases_and_multilingual_queries_preserve_text(self):
        for language, canonical, query in (('en', 'en', ' Explain this sentence. '),
            ('English', 'en', 'What is café?'), ('Chinese', 'zh', '請解釋這一句。'),
            ('Japanese', 'ja', 'この文は？'), ('fr', 'fr', 'Que signifie été ?'),
            ('ar', 'ar', 'ما معنى هذا؟'), ('hi', 'hi', 'इसका अर्थ क्या है?')):
            with self.subTest(language=language):
                provider = Provider(json.dumps(dict(user_language=language,
                    standardized_query=query, extra=[1, 2.5, None])))
                result = QuestionStandardizer(provider).standardize(MARKER, 'English')
                self.assertEqual(result.original_query, MARKER)
                self.assertEqual(result.standardized_query, query)
                self.assertEqual(result.user_language.value, canonical)
                self.assertEqual(result.document_language, LanguageCode.EN)
                self.assertEqual(len(provider.calls), 1)

    def test_provider_failure_is_safe_without_retry(self):
        for failure in (TimeoutError(MARKER), ConnectionError(MARKER), ValueError(MARKER)):
            with self.subTest(kind=type(failure).__name__):
                error = self.reject(failure)
                self.assertEqual(str(error), 'question_standardization_provider_failed')

    def test_exception_logging_and_all_streams_omit_content_and_original_chain(self):
        replies = (MARKER, '{"user_language":"en","'+MARKER+'":true}',
                   '{"user_language":"'+MARKER+'","standardized_query":"Q"}',
                   TimeoutError(MARKER))
        for reply in replies:
            with self.subTest(kind=type(reply).__name__):
                provider = Provider(reply)
                captured = io.StringIO()
                logger = logging.getLogger('question_boundary_capture')
                handler = logging.StreamHandler(captured)
                logger.addHandler(handler)
                try:
                    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                        try:
                            QuestionStandardizer(provider).standardize(MARKER, 'en')
                        except Exception:
                            logger.exception('question_stage_failed')
                        else:
                            self.fail('invalid output accepted')
                finally:
                    logger.removeHandler(handler)
                self.assertNotIn(MARKER, captured.getvalue())
                self.assertIn('question_standardization_', captured.getvalue())
                self.assertEqual(len(provider.calls), 1)

    def test_invalid_outputs_stop_actual_qa_before_scope_retrieval_relevance_answer(self):
        replies = [json.dumps(dict(user_language='en', standardized_query=value))
                   for value in ([], {}, 12, True, ' \n', None)]
        replies += ['[]', MARKER, '{"user_language":"en","standardized_query":"Q","extra":NaN}',
                    '{"user_language":"unknown","standardized_query":"Q"}', TimeoutError(MARKER)]
        for reply in replies:
            with self.subTest(reply=reply):
                coordinator, provider, bundle, relevance, scope = pipeline(reply)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(Exception) as caught:
                        coordinator.ask(MARKER, 'book')
                    self.assertIsInstance(caught.exception, (ValueError, RuntimeError))
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(bundle.search_calls, [])
                scope.resolve.assert_not_called()
                relevance.evaluate.assert_not_called()
                coordinator.prompt_assembler.build_answer_prompt.assert_not_called()
                self.assertEqual(coordinator.llm_provider.calls, [])

    def test_valid_context_diagnostics_omit_text_but_retrieval_retains_exact_query(self):
        query = 'Explain this sentence: ' + MARKER
        provider = Provider(json.dumps(dict(user_language='en', standardized_query=query)))
        bundle = Bundle()
        resolver = QuestionScopeResolver(QuestionScopeKeywordsProvider(), Embedder(query, .9),
            EmbeddingSimilarityService(), llm_fallback_enabled=False)
        relevance = SimpleNamespace(evaluate=lambda results: AnswerMode(AnswerLevel.STRICT, 'fixture'))
        context = ContextOrchestrator(QuestionStandardizer(provider), relevance, resolver,
            Coverage(), Budget(), Builder())
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            result = context.build(query, bundle, top_k=3, session_active_chunk_index=10)
        self.assertNotIn(MARKER, captured.getvalue())
        self.assertNotIn(MARKER, repr(result.standardized_question))
        self.assertEqual(result.standardized_question.original_query, query)
        self.assertEqual(result.standardized_question.standardized_query, query)
        self.assertEqual(bundle.search_calls[0]['query'], query)
        self.assertEqual(len(provider.calls), 1)

    def test_http_invalid_output_details_are_safe_and_no_downstream_dispatch(self):
        for reply, status in ((MARKER, 500), ('{"user_language":"en","private":"'+MARKER+'"}', 500),
            ('{"user_language":"'+MARKER+'","standardized_query":"Q"}', 400),
            (TimeoutError(MARKER), 500), ('[]', 500)):
            with self.subTest(status=status, reply=reply):
                coordinator, provider, bundle, relevance, scope = pipeline(reply)
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), TestClient(route_app(coordinator)) as client:
                    response = client.post('/documents/ask', json=dict(doc_name='book', query=MARKER, session_id='fixed'))
                self.assertEqual(response.status_code, status)
                self.assertNotIn(MARKER, response.text + captured.getvalue())
                self.assertTrue(response.json()['detail'].startswith('question_standardization_'))
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(bundle.search_calls, [])
                relevance.evaluate.assert_not_called()
                self.assertEqual(coordinator.llm_provider.calls, [])

    def scope(self, reply):
        query = 'How did events unfold?'
        provider = Provider(reply)
        resolver = QuestionScopeResolver(QuestionScopeKeywordsProvider(), Embedder(query, .6),
            EmbeddingSimilarityService(), provider, llm_summary_char_limit=12)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            result = resolver.resolve(StandardizedQuestion(query, query, LanguageCode.EN, LanguageCode.EN),
                document_summary='ABCDEFGHIJKLMNOPQRST')
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(provider.calls[0].endswith('document_summary: ABCDEFGHIJKL\n'))
        self.assertNotIn(MARKER.casefold(), (captured.getvalue() + result.method).casefold())
        return result, captured.getvalue()

    def test_unusable_scope_falls_local_with_safe_fixed_diagnostics(self):
        for reply in (MARKER, '{}', '[]', '[{"scope":"global"}]', 'null', None, '{"scope":"'+MARKER+'"}',
                      '{"scope":["global"]}', '{"scope":true}',
                      '{"scope":"global","scope":"local"}',
                      '{"scope":"global","reason":NaN}', TimeoutError(MARKER)):
            with self.subTest(reply=reply):
                result, log = self.scope(reply)
                self.assertEqual((result.scope, result.method), (QuestionScope.LOCAL, 'fallback_local'))
                self.assertIn('llm_scope_', log)

    def test_valid_scope_and_legacy_json_blocks_ignore_arbitrary_reason(self):
        for scope in ('global', 'local'):
            for reason in (MARKER, {'secret': MARKER}, None, [MARKER]):
                for wrapper in ('{}', '```json\n{}\n```'):
                    with self.subTest(scope=scope, reason=reason, wrapper=wrapper):
                        result, log = self.scope(wrapper.format(json.dumps(dict(scope=scope, reason=reason))))
                        self.assertEqual((result.scope, result.method), (QuestionScope(scope), 'llm_fallback'))
                        self.assertEqual(log, '')


if __name__ == '__main__':
    unittest.main()
