"""QUESTION-05/06: requested range and complete scope-envelope integration."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest

from language.language_code import LanguageCode
from question.standardized.standardized_question import StandardizedQuestion
from test_question_output_boundary import Provider
from test_question_scope_routing import (
    AnswerLevel, AnswerMode, Budget, Builder, Bundle, ContextOrchestrator, Coverage,
    Embedder, EmbeddingSimilarityService, QuestionScopeKeywordsProvider,
    QuestionScopeResolver, QuestionStandardizer,
)

EXTENTS = (
    ('List all chapters that discuss this sentence.', 'global'),
    ('Summarize all sections related to this paragraph.', 'global'),
    ('Compare this paragraph with every chapter in the book.', 'global'),
    ('Explain this sentence, not the whole book.', 'local'),
    ('What does "whole book" mean in this sentence?', 'local'),
    ('Explain the phrase "entire document" in this paragraph.', 'local'),
    ('List all verbs in this sentence.', 'local'),
    ('Summarize the whole book, including this paragraph.', 'global'),
    ('Compare this paragraph with themes across the document.', 'global'),
)


QUOTED_REQUESTS = (
    ('Summarize the "whole book", including this paragraph.', 'global'),
    ('Compare this paragraph with the "whole book".', 'global'),
    ('Summarize the whole book, including this paragraph.', 'global'),
    ('Compare this paragraph with the whole book.', 'global'),
    ('What does "whole book" mean in this sentence?', 'local'),
    ('Explain the phrase "entire document" in this paragraph.', 'local'),
    ('Explain this sentence, not the whole book.', 'local'),
    ('Explain this sentence, not the "whole book".', 'local'),
    ('Compare this paragraph with this sentence, not with the "whole book".', 'local'),
)


def resolver(query, reply=None, score=0.0):
    provider = Provider(reply)
    embedder = Embedder(query, score)
    scope = QuestionScopeResolver(QuestionScopeKeywordsProvider(), embedder,
        EmbeddingSimilarityService(), provider)
    return scope, provider, embedder


def context_route(query, language='en', reply=None, score=0.0):
    standardizer = Provider(json.dumps(dict(user_language=language, standardized_query=query)))
    scope, provider, embedder = resolver(query, reply, score)
    builder, coverage, bundle = Builder(), Coverage(), Bundle()
    bundle.document_language = language
    context = ContextOrchestrator(QuestionStandardizer(standardizer),
        SimpleNamespace(evaluate=lambda results: AnswerMode(AnswerLevel.STRICT, 'fixture')),
        scope, coverage, Budget(), builder)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = context.build(query, bundle, top_k=3, session_active_chunk_index=10)
    return result, bundle, builder, coverage, standardizer, provider, captured.getvalue()


class RequestedScopeTests(unittest.TestCase):
    def assert_route(self, query, expected, language='en', reply=None, score=0.0, scope_calls=0):
        result, bundle, builder, coverage, std, provider, log = context_route(query, language, reply, score)
        self.assertEqual(result.metadata['scope'], expected)
        self.assertEqual(bundle.search_calls, [dict(query=query, query_type='str', top_k=8 if expected=='global' else 3)])
        self.assertEqual(builder.local_calls, int(expected=='local'))
        self.assertEqual(coverage.calls, int(expected=='global'))
        self.assertEqual(result.standardized_question.original_query, query)
        self.assertEqual(result.standardized_question.standardized_query, query)
        self.assertEqual(len(std.calls), 1)
        self.assertEqual(len(provider.calls), scope_calls)
        self.assertEqual(result.mode.value, 'local_window_mode' if expected=='local' else 'retrieval_mode')
        return result, log

    def test_quoted_request_roles_resolver(self):
        for query, expected in QUOTED_REQUESTS:
            for score in (0.0, .99):
                with self.subTest(query=query, score=score):
                    scope, provider, embedder = resolver(query, score=score)
                    question = StandardizedQuestion(query, query, LanguageCode.EN, LanguageCode.EN)
                    result = scope.resolve(question, session_active_chunk_index=10)
                    self.assertEqual(result.scope.value, expected)
                    self.assertEqual((question.original_query, question.standardized_query), (query, query))
                    self.assertEqual((provider.calls, embedder.calls), ([], []))

    def test_quoted_request_roles_context(self):
        for query, expected in QUOTED_REQUESTS:
            for score in (0.0, .99):
                with self.subTest(query=query, score=score):
                    self.assert_route(query, expected, score=score)

    def test_quoted_roles_variants_and_mentions_do_not_grant_request_authority(self):
        for left, right in (('"','"'), ("'","'"), ('“','”'), ('‘','’'), ('«','»'), ('「','」'), ('『','』')):
            quoted = left + 'whole book' + right
            for query, expected in (
                (f'Summarize the {quoted}, including this paragraph.', 'global'),
                (f'Please compare this paragraph with the {quoted}.', 'global'),
                (f'Explain {quoted} in this sentence.', 'local'),
                (f'Explain this sentence, not the {quoted}.', 'local'),
                (f'Compare this paragraph with this sentence rather than with the {quoted}.', 'local'),
                (f'Explain "compare" in this sentence with the {quoted}.', 'local'),
            ):
                with self.subTest(query=query):
                    self.assert_route(query, expected)
        for query in (
            'Please summarize the "entire document", including this paragraph.',
            'SUMMARIZE\tTHE\n"whole book", including this paragraph.',
            'Compare this paragraph to the "whole book".',
        ):
            self.assert_route(query, 'global')

    def test_demonstrated_extent_resolver_both_directions(self):
        for query, expected in EXTENTS:
            for score in (0.0, .99):
                with self.subTest(query=query, score=score):
                    scope, provider, embedder = resolver(query, score=score)
                    question = StandardizedQuestion(query, query, LanguageCode.EN, LanguageCode.EN)
                    result = scope.resolve(question, session_active_chunk_index=10)
                    self.assertEqual(result.scope.value, expected)
                    self.assertEqual(question.original_query, query)
                    self.assertEqual(question.standardized_query, query)
                    self.assertEqual(provider.calls, [])
                    self.assertEqual(embedder.calls, [])

    def test_demonstrated_extent_context_both_directions(self):
        for query, expected in EXTENTS:
            with self.subTest(query=query):
                self.assert_route(query, expected, score=.99)

    def test_each_range_occurrence_retains_its_own_exclusion(self):
        for query, expected in (
            ('Explain this sentence, rather than the whole book.', 'local'),
            ('Explain this paragraph instead of the entire document.', 'local'),
            ('Compare this paragraph, not the whole book, with every chapter.', 'global'),
            ('Explain "whole book" in this sentence, then summarize the whole book.', 'global'),
        ):
            with self.subTest(query=query):
                self.assert_route(query, expected)

    def test_quote_pairs_whitespace_and_word_boundaries(self):
        for left, right in (('"','"'), ("'","'"), ('“','”'), ('‘','’'), ('«','»'), ('「','」'), ('『','』')):
            with self.subTest(quote=left):
                self.assert_route(f'Explain {left}whole book{right} in this sentence.', 'local')
        for gap in (' ', '  ', '\n', '\t', '\u00a0'):
            with self.subTest(gap=repr(gap)):
                self.assert_route('List all'+gap+'chapters related to this sentence.', 'global')
                self.assert_route('Explain this sentence, not'+gap+'the'+gap+'whole book.', 'local')
        for query in ('Explain small chapters in this sentence.', 'Explain every chaptered word in this sentence.'):
            self.assert_route(query, 'local')

    def test_multilingual_ranges_quotes_and_adjacent_exclusions(self):
        for language, query, expected in (
            ('zh', '列出所有章節中與這句有關的內容。', 'global'),
            ('zh', '解釋這句，而不是全書。', 'local'),
            ('zh', '解釋這句中的「全書」。', 'local'),
            ('fr', 'Compare ce paragraphe avec tous les chapitres.', 'global'),
            ('fr', 'Explique ce paragraphe, pas tout le livre.', 'local'),
            ('fr', 'Explique « tout le livre » dans ce paragraphe.', 'local'),
            ('ja', 'この文に関係するすべての章を要約。', 'global'),
            ('ja', '本全体ではなく、この文を説明。', 'local'),
            ('ko', '이 문장과 모든 장을 비교하세요.', 'global'),
            ('ko', '책 전체 말고 이 문장을 설명하세요.', 'local'),
        ):
            with self.subTest(language=language, query=query):
                self.assert_route(query, expected, language)

    def test_nonobject_roots_inside_wrappers_rejected_by_resolver(self):
        roots = ('[{"scope":"global"}]', '[[{"scope":"global"}]]', 'null', 'true', '42', '"global"')
        wrappers = ('{}', '```json\n{}\n```', 'Result: {}', '結果：{}')
        for root in roots:
            for wrapper in wrappers:
                with self.subTest(root=root, wrapper=wrapper):
                    query = 'How did events unfold?'
                    scope, provider, _ = resolver(query, wrapper.format(root), .6)
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        result = scope.resolve(StandardizedQuestion(query, query, LanguageCode.EN, LanguageCode.EN))
                    self.assertEqual((result.scope.value, result.method), ('local', 'fallback_local'))
                    self.assertEqual(captured.getvalue().strip(), 'Warn:QuestionScopeResolver#resolve: llm_scope_parse_failed')
                    self.assertEqual(len(provider.calls), 1)

    def test_complete_envelopes_at_actual_context_boundary(self):
        for root, expected in (('[{"scope":"global"}]','local'), ('{"scope":"global"}','global')):
            for wrapper in ('{}','```json\n{}\n```','Result: {}','結果：{}'):
                with self.subTest(root=root, wrapper=wrapper):
                    result, log = self.assert_route('How did events unfold?', expected,
                        reply=wrapper.format(root), score=.6, scope_calls=1)
                    self.assertEqual(result.metadata['scope_method'], 'llm_fallback' if expected=='global' else 'fallback_local')
                    self.assertEqual('llm_scope_parse_failed' in log, expected=='local')

    def test_duplicate_nonfinite_trailing_and_unsupported_envelopes_rejected(self):
        marker = 'PRIVATE_ENVELOPE_MARKER'
        for reply in (
            'Result: {"scope":"global","scope":"local"}',
            '```json\n{"scope":"global","reason":{"x":1,"x":2}}\n```',
            'Result: {"scope":"global","reason":{"x":1e999}}',
            'Result: {"scope":"global","reason":NaN}',
            'Result: [{"scope":"global"}] {"scope":"global"}',
            '```json\n{"scope":"global"}\n``` '+marker,
            'Result: "'+marker+'"',
            marker+' [{"scope":"global"}]',
        ):
            with self.subTest(reply=reply):
                _, log = self.assert_route('How did events unfold?', 'local', reply=reply, score=.6, scope_calls=1)
                self.assertNotIn(marker, log)
                self.assertIn('llm_scope_parse_failed', log)

    def test_valid_nested_objects_and_freeform_reason_remain_safe(self):
        for wrapper in ('{}', '```json\n{}\n```', 'Result: {}', 'Here is the result: {}', '結果：{}'):
            for scope in ('global', 'local'):
                with self.subTest(wrapper=wrapper, scope=scope):
                    reply = wrapper.format(json.dumps(dict(scope=scope, reason={'nested':'PRIVATE_ENVELOPE_MARKER'})))
                    result, log = self.assert_route('How did events unfold?', scope, reply=reply, score=.6, scope_calls=1)
                    self.assertEqual(result.metadata['scope_method'], 'llm_fallback')
                    self.assertNotIn('PRIVATE_ENVELOPE_MARKER', log)


if __name__ == '__main__':
    unittest.main()
