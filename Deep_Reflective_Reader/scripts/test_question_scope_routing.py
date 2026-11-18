"""QUESTION-01/02: real routing with deterministic vector/provider/consumer ports."""
import contextlib,io,json,math,unittest
from types import SimpleNamespace
from context.context_orchestrator import ContextOrchestrator
from embeddings.embedding_similarity_service import EmbeddingSimilarityService
from evaluated_answer.answer_mode import AnswerMode
from question.qa_enums import AnswerLevel,ContextMode
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.question_scope_resolver import QuestionScopeResolver
from question.standardized.question_standardizer import QuestionStandardizer
from retrieval.search_metadata import SearchMetadata
observations=[]
class Provider:
 def __init__(self,reply):self.reply=reply;self.calls=[]
 def complete_text(self,prompt):self.calls.append(prompt);return self.reply
class Embedder:
 def __init__(self,query,score):self.query=query;self.score=score;self.calls=[]
 def get_text_embedding(self,text):
  self.calls.append(text)
  return [1.0,0.0] if text==self.query else [self.score, math.sqrt(1-self.score**2)]
class Budget:
 def compute_available_context_budget(self,**kwargs):return 100
 def join_texts_with_budget(self,*,texts,**kwargs):return ('\n'.join(texts),len(texts),False)
class Builder:
 def __init__(self):self.local_calls=0
 def estimate_full_text_tokens(self,**kwargs):return 10000
 def build_local_window_dynamic(self,**kwargs):
  self.local_calls+=1;return(['FOCUSED_LOCAL_WINDOW'],1,False,1)
class Coverage:
 def __init__(self):self.calls=0
 def select_for_global_scope(self,*,results,**kwargs):
  self.calls+=1;return SimpleNamespace(selected_results=results,raw_chunk_indices=[10],selected_chunk_indices=[10])
class Bundle:
 document_language='en';profile=SimpleNamespace(summary='A source summary');max_context_tokens=100;max_prompt_tokens=200;reserved_output_tokens=20;model_capabilities=None
 def __init__(self):self.search_calls=[];self.id_to_record={1:SimpleNamespace(chunk_index=lambda:10)}
 def search(self,query,top_k):
  self.search_calls.append(dict(query=query,query_type=type(query).__name__,top_k=top_k))
  return [SearchMetadata(1,'n1','RETRIEVED_CHUNK',.5)]
 def get_node_by_id(self,idx):return self.id_to_record.get(idx)
def exercise(query,score=0.0,reply=None):
 provider=Provider(json.dumps(reply if reply is not None else dict(user_language='en',standardized_query=query)))
 emb=Embedder(query,score);resolver=QuestionScopeResolver(QuestionScopeKeywordsProvider(),emb,EmbeddingSimilarityService(),llm_fallback_enabled=False)
 builder=Builder();coverage=Coverage();bundle=Bundle()
 orchestrator=ContextOrchestrator(QuestionStandardizer(provider),SimpleNamespace(evaluate=lambda results:AnswerMode(AnswerLevel.STRICT,'fixture match')),resolver,coverage,Budget(),builder)
 buf=io.StringIO();failure=None;result=None
 with contextlib.redirect_stdout(buf):
  try: result=orchestrator.build(query=query,bundle=bundle,top_k=3,session_active_chunk_index=10)
  except Exception as e:failure=dict(type=type(e).__name__,message=str(e))
 row=dict(query=query,semantic_score=score,error=failure,mode=result.mode.value if result else None,scope=result.metadata['scope'] if result else None,method=result.metadata['scope_method'] if result else None,context=result.context_text if result else None,search_calls=bundle.search_calls,local_calls=builder.local_calls,coverage_calls=coverage.calls,provider_calls=len(provider.calls),embedding_calls=len(emb.calls))
 observations.append(row);return row

from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from question.qa_enums import QuestionScope, LocalSignalQuality
from question.standardized.standardized_question import StandardizedQuestion


class ScopeRoutingTests(unittest.TestCase):
    def resolver(self, text, score=0.0):
        embedder = Embedder(text, score)
        provider = Provider('{"scope":"global","reason":"controlled"}')
        resolver = QuestionScopeResolver(QuestionScopeKeywordsProvider(), embedder,
            EmbeddingSimilarityService(), provider)
        return resolver, embedder, provider

    def resolve(self, text, language='en', score=0.0, active=0):
        resolver, embedder, provider = self.resolver(text, score)
        question = StandardizedQuestion(text, text, LanguageCode(language), LanguageCode(language))
        result = resolver.resolve(question, session_active_chunk_index=active)
        self.assertEqual(question.original_query, text)
        self.assertEqual(question.standardized_query, text)
        return result, embedder, provider

    def test_word_interiors_and_unicode_marks_are_not_cues(self):
        for text, cue in (('small', 'all'), ('listening', 'list'), ('allein', 'alle'),
            ('globalisation', 'global'), ('bana', 'ana'), ('вселенную', 'все'),
            ('where', 'here'), ('كلية', 'كل'), ('मुख्याँ', 'मुख्य'),
            ('all2', 'all'), ('not_all', 'all'), ('áall', 'all'), ('여기저기', '여기')):
            with self.subTest(text=text, cue=cue):
                self.assertIsNone(QuestionScopeResolver._contains_keyword(text, (cue,)))

    def test_true_cues_punctuation_unicode_composition_and_unspaced_scripts(self):
        for text, cue in (('"all",', 'all'), ('Re\u0301sume\u0301', 'résumé'),
            ('請總結全書。', '總結'), ('この段落を要約。', 'この段落'),
            ('อธิบายประโยคนี้ให้ฉัน', 'ประโยคนี้'), ('이 문장을 설명해 주세요.', '이 문장'),
            ('여기에서는?', '여기'), ('هذه الجملة؟', 'هذه الجملة'), ('यह वाक्य?', 'यह वाक्य')):
            with self.subTest(text=text):
                self.assertEqual(QuestionScopeResolver._contains_keyword(text, (cue,)), cue)

    def test_matching_continues_after_an_invalid_interior_occurrence(self):
        self.assertEqual(QuestionScopeResolver._contains_keyword('small and all', ('all',)), 'all')
        self.assertEqual(QuestionScopeResolver._contains_keyword('where is here?', ('here',)), 'here')

    def test_whitespace_is_consistent_for_all_cue_classes(self):
        for separator in (' ', '  ', '\n', '\t', '\u00a0', '\u2003'):
            with self.subTest(separator=repr(separator)):
                result, embedder, provider = self.resolve('Explain this'+separator+'sentence.', score=.9)
                self.assertEqual(result.scope, QuestionScope.LOCAL)
                self.assertEqual(embedder.calls, [])
                self.assertEqual(provider.calls, [])
                result, _, _ = self.resolve('Summarize the whole'+separator+'book, including this paragraph.')
                self.assertEqual(result.scope, QuestionScope.GLOBAL)
                resolver, _, _ = self.resolver('in'+separator+'this'+separator+'part')
                signal = resolver._has_local_reference_signal('in'+separator+'this'+separator+'part', LanguageCode.EN, 0)
                self.assertEqual(signal.quality, LocalSignalQuality.WEAK)

    def test_strong_local_precedes_generic_intent_and_high_similarity_in_all_languages(self):
        for code in LanguageProfileRegistry.get_supported_language_codes():
            language = LanguageCode(code)
            strong = LanguageProfileRegistry.get_strong_local_reference_signals(language)[0]
            generic = QuestionScopeKeywordsProvider.get_keywords(language)[0]
            text = generic+' '+strong
            with self.subTest(language=code):
                result, embedder, provider = self.resolve(text, code, score=.99, active=None)
                self.assertEqual(result.scope, QuestionScope.LOCAL)
                self.assertEqual(result.method, 'strong_local_reference')
                self.assertEqual(embedder.calls, [])
                self.assertEqual(provider.calls, [])

    def test_explicit_document_extent_precedes_local_reference_for_all_languages(self):
        codes = LanguageProfileRegistry.get_supported_language_codes()
        for code in codes:
            language = LanguageCode(code)
            broad = QuestionScopeKeywordsProvider.get_document_extent_signals(language)
            self.assertTrue(broad, code)
            local = LanguageProfileRegistry.get_strong_local_reference_signals(language)[0]
            for phrase in broad:
                with self.subTest(language=code, phrase=phrase):
                    result, embedder, provider = self.resolve(local+' / '+phrase, code)
                    self.assertEqual(result.scope, QuestionScope.GLOBAL)
                    self.assertEqual(result.matched_keyword, phrase)
                    self.assertEqual(embedder.calls, [])
                    self.assertEqual(provider.calls, [])
        combined = QuestionScopeKeywordsProvider.get_document_extent_signals(LanguageCode.UNKNOWN)
        self.assertEqual(set(combined), {cue for code in codes for cue in QuestionScopeKeywordsProvider.get_document_extent_signals(LanguageCode(code))})

    def test_local_quantifiers_do_not_mean_whole_document(self):
        for text in ('List all verbs in this sentence.', 'Summarize this whole paragraph.',
                     'Which themes appear in this paragraph?', 'Explain this sentence.'):
            with self.subTest(text=text):
                result, _, _ = self.resolve(text, score=.99)
                self.assertEqual(result.scope, QuestionScope.LOCAL)

    def test_weak_anchors_still_depend_on_active_session_and_valid_boundaries(self):
        for text in ('here', 'where'):
            for active in (None, 0):
                with self.subTest(text=text, active=active):
                    resolver, _, _ = self.resolver(text)
                    signal = resolver._has_local_reference_signal(text, LanguageCode.EN, active)
                    self.assertEqual(signal.matched, text=='here' and active is not None)
                    if signal.matched:self.assertEqual(signal.quality, LocalSignalQuality.WEAK)

    def test_context_local_routes_preserve_requested_top_k_and_source_query(self):
        for query in ('What does small mean in this sentence?', 'Why is he listening in this paragraph?',
                      'Which word in this sentence is a noun?', 'Summarize this paragraph.',
                      'List all verbs in this sentence.', 'Explain this\nsentence.'):
            with self.subTest(query=query):
                row = exercise(query, score=.9)
                self.assertIsNone(row['error'])
                self.assertEqual(row['mode'], ContextMode.LOCAL_WINDOW.value)
                self.assertEqual(row['scope'], 'local')
                self.assertEqual(row['local_calls'], 1)
                self.assertEqual(row['coverage_calls'], 0)
                self.assertEqual(row['search_calls'], [dict(query=query, query_type='str', top_k=3)])
                self.assertEqual(row['provider_calls'], 1)

    def test_context_explicit_broad_routes_still_use_coverage(self):
        for query in ('Summarize the whole book.', 'Compare this paragraph with themes across the document.',
                      'Summarize the whole book, including this paragraph.'):
            with self.subTest(query=query):
                row = exercise(query)
                self.assertIsNone(row['error'])
                self.assertEqual(row['scope'], 'global')
                self.assertEqual(row['coverage_calls'], 1)
                self.assertEqual(row['local_calls'], 0)
                self.assertEqual(row['search_calls'][0]['top_k'], 8)
                self.assertEqual(row['provider_calls'], 1)


if __name__ == '__main__':
    unittest.main()
