"""Real file summary publication with actual SDK status, isolated sources."""

import contextlib, io, socket, tempfile, unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_llm_completion_admission import payload
import httpx
from openai import OpenAI
from llm.openai_llm_provider import OpenAILLMProvider
from app.section_task_coordinator import SectionTaskCoordinator
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from test_reading_interaction_artifact_store import _document
from test_summary_persistence_cache import (
    _FakePreparationPipeline,
    _FakeProfileStore,
    _FakeTaskUnitResolver,
    _FakeEnhancedParseEvaluator,
)
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.section_task_context_builder import SectionTaskContextBuilder
from section_tasks.section_task_prompt_builder_factory import (
    SectionTaskPromptBuilderFactory,
)
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon
from section_tasks.topic_guidance_registry import TopicGuidanceRegistry
from section_tasks.summary_task_prompt_builder import SummaryTaskPromptBuilder
from section_tasks.section_quiz_task_prompt_builder import SectionQuizTaskPromptBuilder
from section_tasks.chapter_quiz_task_prompt_builder import ChapterQuizTaskPromptBuilder

BODY = "This passage explains why clear assumptions support sound conclusions. " * 12
GOOD = "A complete accepted summary with both the assumptions and conclusions."
PARTIAL = "The chapter describes the first assumption, but the remaining explanation"
rows = []


@contextlib.contextmanager
def setup(model):
    with tempfile.TemporaryDirectory() as directory:
        repo = StructuredDocumentArtifactRepository(base_dir=directory)
        doc = _document()
        c = doc.chapters[0]
        s = c.sections[0]
        s = replace(
            s,
            content=BODY,
            char_start=0,
            char_end=len(BODY),
            task_units=[replace(s.task_units[0], content=BODY)],
        )
        repo.save_reparsed_document(
            replace(
                doc,
                document_id="book",
                raw_text=BODY,
                chapters=[replace(c, sections=[s])],
            ),
            "book",
        )
        state = {"mode": "completed", "text": GOOD}
        calls = []

        def response(request):
            calls.append({"mode": state["mode"], "path": request.url.path})
            return httpx.Response(
                200, json=payload(model, state["mode"], state["text"])
            )

        p = OpenAILLMProvider(
            SimpleNamespace(get=lambda: "synthetic-invalid-key"), model=model
        )
        old = getattr(p.llm, "_client", None)
        if old is not None:
            old.close()
        client = OpenAI(
            api_key="synthetic-invalid-key",
            base_url="https://offline.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        )
        p.llm._client = client
        p.llm.api_base = "https://offline.invalid/v1"
        common = SectionTaskPromptCommon(TopicGuidanceRegistry())
        factory = SectionTaskPromptBuilderFactory(
            SummaryTaskPromptBuilder(common),
            SectionQuizTaskPromptBuilder(common),
            ChapterQuizTaskPromptBuilder(common),
        )
        resolver = _FakeTaskUnitResolver()
        context = SectionTaskContextBuilder()
        co = SectionTaskCoordinator(
            _FakePreparationPipeline(repo),
            repo,
            _FakeProfileStore(),
            ChapterSummaryService(p, context, factory, resolver),
            ChapterQuizService(p, context, factory, resolver),
            resolver,
            _FakeEnhancedParseEvaluator(),
        )
        co.ensure_document_task_layout(doc_name="book")
        try:
            yield repo, co, state, calls
        finally:
            client.close()


def saved(repo):
    section = repo.load_document("book").chapters[0].sections[0]
    return (
        section.task_artifacts.summary.content
        if section.task_artifacts and section.task_artifacts.summary
        else None
    )


class Publication(unittest.TestCase):
    def setUp(self):
        guard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        guard.start()
        self.addCleanup(guard.stop)

    def test_incomplete_initial_output_is_not_published(self):
        for model in ("gpt-4.1-mini", "gpt-4o-mini"):
            with self.subTest(model=model), setup(model) as (repo, co, state, calls):
                raw_before = repo.load_document("book").raw_text
                state.update(mode="incomplete", text=PARTIAL)
                result = co.summarize_section("book", "section-1")
                persisted = saved(repo)
                rows.append(
                    {
                        "model": model,
                        "case": "initial",
                        "success": result.success,
                        "result": result.payload,
                        "saved": persisted,
                        "calls": len(calls),
                        "raw_unchanged": repo.load_document("book").raw_text
                        == raw_before,
                    }
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(repo.load_document("book").raw_text, raw_before)
                with self.subTest(check="refuse_incomplete"):
                    self.assertFalse(result.success)
                with self.subTest(check="no_artifact"):
                    self.assertIsNone(persisted)

    def test_incomplete_refresh_preserves_valid_summary_and_cache(self):
        for model in ("gpt-4.1-mini", "gpt-4o-mini"):
            with self.subTest(model=model), setup(model) as (repo, co, state, calls):
                valid = co.summarize_section("book", "section-1")
                self.assertTrue(valid.success)
                self.assertEqual(saved(repo), GOOD)
                state.update(mode="incomplete", text=PARTIAL)
                refreshed = co.summarize_section(
                    "book", "section-1", refresh_summary=True
                )
                persisted = saved(repo)
                cached = co.summarize_section("book", "section-1")
                rows.append(
                    {
                        "model": model,
                        "case": "refresh",
                        "success": refreshed.success,
                        "saved": persisted,
                        "cache_hit": cached.cache_hit,
                        "cached": cached.payload,
                        "calls": len(calls),
                    }
                )
                self.assertEqual(len(calls), 2)
                self.assertTrue(cached.cache_hit)
                with self.subTest(check="refuse_incomplete"):
                    self.assertFalse(refreshed.success)
                with self.subTest(check="preserve_saved"):
                    self.assertEqual(persisted, GOOD)
                with self.subTest(check="preserve_cache"):
                    self.assertEqual(cached.payload, GOOD)


if __name__ == "__main__":
    unittest.main()
