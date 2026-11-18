"""CONTEXT-01: real context assembly and QA dispatch, no external providers."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace

import faiss
import numpy as np
from fastapi.testclient import TestClient
from llama_index.core.schema import TextNode

from app.qa_coordinator import QACoordinator
from context.context_orchestrator import ContextOrchestrator
from context.coverage_oriented_context_builder import CoverageOrientedContextBuilder
from context.document_context_builder import DocumentContextBuilder
from context.token_budget_manager import TokenBudgetManager
from embeddings.embedding_similarity_service import EmbeddingSimilarityService
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator
from language.language_code import LanguageCode
from llm.llm_model_capabilities import LLMModelCapabilities
from profile.document_profile import DocumentProfile
from prompts.prompt_assembler import PromptAssembler
from question.qa_enums import ContextMode
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.question_scope_resolver import QuestionScopeResolver
from question.standardized.question_standardizer import QuestionStandardizer
from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata
from session.session_manager import SessionManager, SessionUpdateResult
from scripts.test_question_output_boundary import route_app


class FixedProvider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete_text(self, prompt):
        self.calls.append(prompt)
        return self.reply


class FixedEmbedder:
    def get_text_embedding(self, text):
        return [1.0, 0.0]


class CapturedContext:
    """Observe actual results without substituting any assembly decisions."""
    def __init__(self, actual):
        self.actual = actual
        self.result = None

    def build(self, **kwargs):
        self.result = self.actual.build(**kwargs)
        return self.result


def make_case(query="Explain this sentence.", texts=None):
    assembler = PromptAssembler()
    budget = TokenBudgetManager(assembler)
    builder = DocumentContextBuilder(budget)
    texts = texts if texts is not None else ["A" * 5000, "B" * 5000]
    records = {}
    offset = 0
    for i, text in enumerate(texts):
        records[i] = NodeRecord(TextNode(
            text=text, id_=str(i), metadata={
                "chunk_index": i, "char_start": offset, "char_end": offset + len(text),
            },
        ), i)
        offset += len(text)
    index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
    vectors = np.array([[1.0, 0.0]] + [[0.0, 1.0]] * (len(texts) - 1), dtype="float32")
    index.add_with_ids(vectors, np.arange(len(texts), dtype="int64"))
    bundle = FaissIndexBundle(
        index, FixedEmbedder(), LLMModelCapabilities("controlled", "chat_completions", 16000, 1000),
        records, 2, "en", 1500, 3200, 500,
    )
    bundle.set_profile(DocumentProfile("Synthetic", "Synthetic source summary", LanguageCode.EN))
    standardizer = FixedProvider(json.dumps({"user_language": "en", "standardized_query": query}))
    scope_provider = FixedProvider('{"scope":"local"}')
    resolver = QuestionScopeResolver(
        QuestionScopeKeywordsProvider(), FixedEmbedder(), EmbeddingSimilarityService(),
        llm_provider=scope_provider, llm_fallback_enabled=False,
    )
    actual = ContextOrchestrator(
        QuestionStandardizer(standardizer), QuestionRelevanceEvaluator(), resolver,
        CoverageOrientedContextBuilder(), budget, builder,
    )
    captured = CapturedContext(actual)
    answer = FixedProvider("Controlled answer")
    coordinator = object.__new__(QACoordinator)
    coordinator.document_preparation_pipeline = SimpleNamespace(prepare_and_load=lambda name: SimpleNamespace(
        bundle=bundle, assets=SimpleNamespace(
            structured_document_path=None, bundle_ready=True, faiss_ready=True, profile_ready=True,
        ),
    ))
    coordinator.context_orchestrator = captured
    coordinator.prompt_assembler = assembler
    coordinator.llm_provider = answer
    coordinator.session_manager = SessionManager()
    return SimpleNamespace(
        query=query, bundle=bundle, budget=budget, builder=builder, coordinator=coordinator,
        context=captured, answer=answer, standardizer=standardizer, scope_provider=scope_provider,
    )


def seed_session_anchor(case):
    manager = case.coordinator.session_manager
    snapshot = manager.get_or_create_session("isolated", "synthetic")
    manager.update_session(snapshot, SessionUpdateResult(
        question="seed", bundle=case.bundle,
        results=[SearchMetadata(0, "0", "controlled source", 0.0)],
    ))


def ask(case, active=False):
    session_id = "isolated" if active else None
    if active:
        seed_session_anchor(case)
    with contextlib.redirect_stdout(io.StringIO()):
        return case.coordinator.ask(case.query, "synthetic", session_id=session_id)


class ContextCapacityTests(unittest.TestCase):
    def test_join_none_zero_negative_and_positive(self):
        manager = TokenBudgetManager(PromptAssembler())
        for limit in (0, -1, -100):
            with self.subTest(limit=limit):
                self.assertEqual(manager.join_texts_with_budget(["Evidence"], 1500, limit), ("", 0, True))
                self.assertEqual(manager.join_texts_with_budget([], 1500, limit), ("", 0, False))
        self.assertEqual(manager.join_texts_with_budget(["Evidence"], 1500, None), ("Evidence", 2, False))
        self.assertEqual(manager.join_texts_with_budget(["Evidence", "Next"], 1500, 1), ("Evid", 1, True))
        self.assertEqual(manager.join_texts_with_budget(["Evidence"], 0, None), ("", 0, True))

    def test_local_and_full_helpers_preserve_budget_and_order(self):
        case = make_case(texts=["LEFT FOUR", "CENTER", "RIGHT FOUR"])
        case.bundle.max_context_tokens = 8
        # Different insertion order must not change physical source order.
        case.bundle.id_to_record = dict(reversed(list(case.bundle.id_to_record.items())))
        for limit in (0, -1, -100):
            with self.subTest(limit=limit):
                self.assertEqual(case.builder.build_local_window_dynamic(case.bundle, 1, limit), ([], 0, True, 0))
                self.assertEqual(case.builder.build_full_text_context(case.bundle, limit), ("", 0, True))
        self.assertEqual(case.builder.build_local_window_dynamic(case.bundle, 1),
                         (["LEFT FOUR", "CENTER", "RIGHT FOUR"], 7, False, 1))
        self.assertEqual(case.builder.build_local_window_dynamic(case.bundle, 1, 2), (["CENTER"], 2, True, 0))
        self.assertEqual(case.builder.build_local_window_dynamic(case.bundle, 1, 1), (["CENT"], 1, True, 0))
        self.assertEqual(case.builder.build_full_text_context(case.bundle), ("LEFT FOUR\nCENTER\nRIGHT FOUR", 7, False))
        self.assertEqual(case.builder.build_full_text_context(case.bundle, 2), ("LEFT FOUR", 2, True))

    def assert_capacity_refused(self, case, active):
        with self.assertRaisesRegex(RuntimeError, "^context_capacity_exceeded$") as caught:
            ask(case, active)
        self.assertEqual(type(caught.exception).__name__, "ContextCapacityError")
        self.assertEqual(case.answer.calls, [])
        self.assertEqual(len(case.standardizer.calls), 1)
        self.assertEqual(case.scope_provider.calls, [])
        self.assertIsNone(case.context.result)
        if active:
            session = case.coordinator.session_manager.get_session("isolated")
            self.assertEqual(session.active_chunk_index, 0)

    def test_long_question_refuses_retrieval_and_local_answer(self):
        for active in (False, True):
            with self.subTest(active=active):
                case = make_case("Explain this sentence. PRIVATE_CAPACITY_MARKER" + " question" * 1000)
                self.assert_capacity_refused(case, active)

    def test_capacity_failure_remains_safe_through_actual_ask_route(self):
        marker = "PRIVATE_CAPACITY_MARKER"
        for active in (False, True):
            with self.subTest(active=active):
                case = make_case("Explain this sentence. " + marker + " question" * 1000)
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    if active:
                        seed_session_anchor(case)
                    with TestClient(route_app(case.coordinator)) as client:
                        response = client.post("/documents/ask", json={
                            "doc_name": "synthetic", "query": case.query, "session_id": "isolated",
                        })
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json(), {"detail": "context_capacity_exceeded"})
                self.assertNotIn(marker, response.text + captured.getvalue())
                self.assertEqual(case.answer.calls, [])
                self.assertEqual(len(case.standardizer.calls), 1)
                self.assertEqual(case.scope_provider.calls, [])

    def test_fixed_prompt_exact_exhaustion_and_one_token_below(self):
        for active in (False, True):
            for spare in (0, -1):
                with self.subTest(active=active, spare=spare):
                    case = make_case()
                    ask(case, active)
                    result = case.context.result
                    fixed = case.budget.estimate_non_context_prompt_tokens(
                        result.standardized_question, result.answer_mode, result.prompt_mode, case.bundle.profile,
                    )
                    case.bundle.max_prompt_tokens = fixed + case.bundle.reserved_output_tokens + spare
                    case.answer.calls.clear()
                    case.standardizer.calls.clear()
                    case.context.result = None
                    self.assert_capacity_refused(case, active)

    def test_one_remaining_token_clips_center_and_dispatches_once(self):
        for active in (False, True):
            with self.subTest(active=active):
                case = make_case()
                ask(case, active)
                result = case.context.result
                fixed = case.budget.estimate_non_context_prompt_tokens(
                    result.standardized_question, result.answer_mode, result.prompt_mode, case.bundle.profile,
                )
                case.bundle.max_prompt_tokens = fixed + case.bundle.reserved_output_tokens + 1
                case.answer.calls.clear()
                ask(case, active)
                self.assertEqual(case.context.result.metadata["budget"], 1)
                self.assertEqual(case.context.result.context_text, "AAAA")
                self.assertEqual(len(case.answer.calls), 1)
                self.assertLessEqual(case.budget.estimate_tokens(case.answer.calls[0]) + 500, case.bundle.max_prompt_tokens)

    def test_positive_retrieval_and_local_keep_routing_and_source(self):
        for active, mode in ((False, ContextMode.RETRIEVAL), (True, ContextMode.LOCAL_WINDOW)):
            with self.subTest(active=active):
                case = make_case()
                ask(case, active)
                result = case.context.result
                self.assertEqual(result.mode, mode)
                self.assertEqual(result.metadata["budget"], 1500)
                self.assertEqual(result.metadata["token_used"], 1250)
                self.assertEqual(result.context_text, "A" * 5000)
                self.assertEqual(result.standardized_question.original_query, case.query)
                self.assertEqual(len(case.answer.calls), 1)
                self.assertEqual(len(case.standardizer.calls), 1)
                self.assertEqual(case.scope_provider.calls, [])
                self.assertLessEqual(case.budget.estimate_tokens(case.answer.calls[0]) + 500, 3200)

    def test_full_text_and_global_positive_controls(self):
        for query in ("Explain this sentence.", "Summarize the whole book."):
            with self.subTest(query=query):
                case = make_case(query, ["Short source.", "Second source."])
                ask(case)
                self.assertEqual(case.context.result.mode, ContextMode.FULL_TEXT)
                self.assertEqual(case.context.result.context_text, "Short source.\nSecond source.")
                self.assertEqual(len(case.answer.calls), 1)
                self.assertEqual(case.scope_provider.calls, [])

    def test_global_retrieval_exhaustion_and_coverage_control(self):
        for long in (False, True):
            with self.subTest(long=long):
                query = "List all chapters." + (" question" * 1000 if long else "")
                case = make_case(query, ["A" * 24000, "B" * 24000])
                if long:
                    self.assert_capacity_refused(case, False)
                else:
                    ask(case)
                    self.assertEqual(case.context.result.mode, ContextMode.RETRIEVAL)
                    self.assertEqual(case.context.result.metadata["scope"], "global")
                    self.assertEqual(case.context.result.metadata["effective_top_k"], 8)
                    self.assertGreater(case.context.result.metadata["global_coverage_evidence_count"], 0)
                    self.assertEqual(len(case.answer.calls), 1)
                    self.assertLessEqual(case.budget.estimate_tokens(case.answer.calls[0]) + 500, 3200)


if __name__ == "__main__":
    unittest.main()
