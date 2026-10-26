"""CLEAN-09 cross-stage contracts; real API/SQL, deterministic model responses.

Run with DEEP_READER_TEST_POSTGRES_DSN against disposable schemas. This is not
live-book/model acceptance. Reuse fixture setup without rediscovering its tests.
"""

from contextlib import contextmanager
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from app.enhancement_operations import EnhancementOperations
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_structure import ocr_cleanup_overlay as overlay
from document_structure.reading_content import ReadingContentConflict, state_for
from document_structure.semantic_structure_plan import read_semantic_prompt_context
from scripts import test_enhancement_cleanup_api as fixtures


KINDS = ("unchanged", "titles_only", "structure_changed")


class CleanupConsistencyTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.CleanupApiTests.setUpClass.__func__)
    drop_schema = classmethod(fixtures.CleanupApiTests.drop_schema.__func__)

    @contextmanager
    def case(self, kind="unchanged"):
        api = fixtures.CleanupApiTests()
        api.scoped = self.scoped
        try:
            api.setUp()
            api.kind = kind
            yield api
        finally:
            api.doCleanups()

    def assert_source_preserved(self, api, before):
        current = api.f.snapshot()[0]
        self.assertEqual(current.raw_text, before.raw_text)
        self.assertEqual("".join(s.content for c in current.chapters for s in c.sections), api.f.text)
        for key in ("extraction", "source_page_spans", "source_review"):
            self.assertEqual(current.parse_provenance[key], before.parse_provenance[key])
        self.assertEqual(api.f.path.read_bytes(), b"immutable fixture, not parsed")
        return current

    def test_real_split_and_cleanup_planners_run_before_single_atomic_commit(self):
        from scripts.test_ocr_cleanup_validation import BODY

        with patch("scripts.test_ocr_cleanup_overlay.BODY", "A unique opening to the body.\n" + BODY), \
                self.case() as api:
            # Remove only the fixture's candidate shortcut, retaining real source
            # resolution, semantic validation, cleanup, storage and layout code.
            api.pipeline.build_enhancement_candidate = (
                DocumentPreparationPipeline.build_enhancement_candidate.__get__(api.pipeline))
            before = api.f.snapshot()[0]
            events = []
            complete = api.provider.complete_text_with_output_budget

            def model(prompt, output_tokens, *, response_schema):
                self.assertEqual(api.f.snapshot()[0], before)
                if "ocr_cleanup_plan_v1" in prompt:
                    events.append("cleanup_model")
                    return complete(prompt, output_tokens, response_schema=response_schema)
                events.append("split_model")
                source = read_semantic_prompt_context(prompt)["source"]["text"]
                self.assertEqual(source, api.f.text)
                return json.dumps({"schema_version": "semantic_structure_plan_v1", "boundaries": [
                    {"start_quote": source[:4], "title": None, "origin": "implicit",
                     "level": 1, "role": "main_body"},
                    {"start_quote": source[6:30], "title": "Body", "origin": "generated",
                     "level": 1, "role": "main_body"}]})

            stage, save = overlay.stage_cleanup, api.f.store.save
            commit, layout = api.f.repo.commit_enhancement, api.coordinator.ensure_document_task_layout

            def staging(*args, **kwargs):
                self.assertEqual(events, ["split_model", "cleanup_model"])
                events.append("stage")
                return stage(*args, **kwargs)

            def saving(*args, **kwargs):
                if kwargs.get("replace_existing_hierarchy"):
                    self.assertEqual(events[-1], "stage")
                    self.assertIsNotNone(kwargs.get("_connection"))
                    events.append("hierarchy_write")
                else:
                    self.assertIn("commit_returned", events)
                return save(*args, **kwargs)

            def committing(*args, **kwargs):
                result = commit(*args, **kwargs)
                events.append("commit_returned")
                return result

            def preparing(*args, **kwargs):
                self.assertEqual(events[-1], "commit_returned")
                events.append("layout")
                return layout(*args, **kwargs)

            request = api.request()
            with patch.object(api.provider, "complete_text_with_output_budget", side_effect=model), \
                    patch.object(overlay, "stage_cleanup", side_effect=staging), \
                    patch.object(api.f.store, "save", side_effect=saving), \
                    patch.object(api.f.repo, "commit_enhancement", side_effect=committing) as commits, \
                    patch.object(api.coordinator, "ensure_document_task_layout", side_effect=preparing), \
                    patch("subprocess.run", side_effect=AssertionError("hidden extraction")):
                result = api.assert_completed(api.execute(request))
            self.assertEqual(result["change_kind"], "structure_changed")
            self.assertEqual(result["cleanup"]["status"], "completed")
            self.assertEqual(commits.call_count, 1)
            self.assertEqual(events, ["split_model", "cleanup_model", "stage", "hierarchy_write",
                                      "commit_returned", "layout"])
            self.assertNotIn("@@??", api.f.content())
            self.assert_source_preserved(api, before)

    def test_missing_malformed_and_empty_cleanup_across_structural_outcomes(self):
        for kind in KINDS:
            for payload, status, reason in (
                    ("", "failed", "invalid_plan"),
                    ('{"schema_version":"ocr_cleanup_plan_v1"}', "failed", "invalid_plan"),
                    ('{"schema_version":"ocr_cleanup_plan_v1","items":[]}', "no_changes", None)):
                with self.subTest(kind=kind, payload=payload), self.case(kind) as api:
                    before = api.f.snapshot()[0]
                    api.provider.payload = payload
                    with patch.object(api.f.repo, "commit_enhancement", wraps=api.f.repo.commit_enhancement) as commit:
                        result = api.assert_completed(api.execute(api.request()))
                    self.assertEqual(commit.call_count, 1)
                    self.assertEqual(result["change_kind"], kind)
                    self.assertEqual(result["cleanup"]["status"], status)
                    self.assertEqual(result["cleanup"]["reason"], reason)
                    self.assertEqual(result["cleanup"]["reading_revision"], 0)
                    self.assertEqual(result["pending_review_count"], 1)
                    self.assertIn("@@??", api.f.content())
                    self.assert_source_preserved(api, before)

    def test_optional_application_failure_retains_uncleaned_candidate(self):
        for kind in KINDS:
            with self.subTest(kind=kind), self.case(kind) as api:
                before = api.f.snapshot()[0]
                stage = overlay.stage_cleanup

                def projection_failure(*args, **kwargs):
                    # Fail only staging of the optional overlay, not later layout.
                    with patch("document_structure.reading_content.project_document",
                               side_effect=ReadingContentConflict("cleanup_projection_failed")):
                        return stage(*args, **kwargs)

                with patch.object(overlay, "stage_cleanup", side_effect=projection_failure):
                    result = api.assert_completed(api.execute(api.request()))
                self.assertEqual(result["change_kind"], kind)
                self.assertEqual(result["cleanup"]["status"], "skipped")
                self.assertEqual(result["cleanup"]["applied_count"], 0)
                self.assertEqual(result["pending_review_count"], 1)
                current = self.assert_source_preserved(api, before)
                self.assertFalse(state_for(current).get("cleanup_spans"))
                self.assertFalse(state_for(current).get("audit"))
                self.assertEqual(len(current.chapters[0].sections), 2 if kind == "structure_changed" else 1)
                if kind == "titles_only":
                    self.assertEqual(current.chapters[0].sections[0].title, "New heading")

    def test_final_receipt_failure_rolls_back_every_structural_outcome(self):
        for kind in KINDS:
            with self.subTest(kind=kind), self.case(kind) as api:
                request, before = api.request(), api.f.snapshot()
                receipt = api.f.store._commit_enhancement_receipt

                def failed_receipt(*args):
                    receipt(*args)
                    raise RuntimeError("injected after SQL writes")

                with patch.object(api.f.store, "_commit_enhancement_receipt", side_effect=failed_receipt), \
                        patch.object(api.coordinator, "ensure_document_task_layout") as layout:
                    self.assertEqual(api.execute(request).status_code, 500)
                layout.assert_not_called()
                self.assertEqual(api.f.snapshot(), before)
                self.assertEqual(api.lookup(request).json()["status"], "unknown")

    def test_lost_ack_restart_failed_layout_then_retry_has_only_one_commit(self):
        with self.case("titles_only") as api:
            request = api.request()
            commit = api.f.repo.commit_enhancement

            def lost(*args, **kwargs):
                commit(*args, **kwargs)
                raise ConnectionError("lost acknowledgement")

            with patch.object(api.f.repo, "commit_enhancement", side_effect=lost) as commits:
                self.assertEqual(api.execute(request).status_code, 500)
                api.service.operations = EnhancementOperations()
                api.service.secret = b"restarted process"
                with patch.object(api.coordinator, "ensure_document_task_layout", side_effect=RuntimeError("layout")):
                    failed = api.execute(request).json()
                    self.assertEqual(failed["phase"], "layout")
                    self.assertEqual(failed["retry_stage"], "layout")
                    self.assertEqual(api.lookup(request).json()["durability"], "persisted_commit")
                result = api.assert_completed(api.execute(request))
                self.assertEqual(api.execute(request).json(), result)
                self.assertEqual(commits.call_count, 1)
            self.assertEqual(len(api.provider.calls), 1)
            api.pipeline.build_enhancement_candidate.assert_called_once()
            state = state_for(api.f.snapshot()[0])
            self.assertEqual(state["revision"], 1)
            self.assertEqual(len(state["audit"]), 1)
            self.assertEqual(len(state["cleanup_receipts"]), 1)

    def test_non_ocr_legacy_opt_out_never_runs_cleanup(self):
        from doc_loaders.extraction_identity import read_extraction_record

        with self.case("titles_only") as api:
            document = api.f.snapshot()[0]
            identity, _ = read_extraction_record(document.parse_provenance)
            document = replace(document, parse_provenance={**document.parse_provenance,
                "extraction": replace(identity, mode="native").record(api.f.text)})
            api.f.repo.save_reparsed_document(document, "book")
            preflight = api.preflight(allow_ocr_cleanup=False).json()
            request = dict(doc_name="book", parser_mode="llm_enhanced", discard_interactions=True,
                           confirmation_token=preflight["confirmation_token"])
            with patch.object(api.pipeline, "prepare_enhancement_cleanup", side_effect=AssertionError("opted out")):
                response = api.execute(request)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["success"])
            self.assertIsNone(response.json().get("cleanup"))
            self.assertEqual(api.provider.calls, [])
            self.assertNotIn("cleanup_receipts", state_for(api.f.snapshot()[0]))


if __name__ == "__main__":
    unittest.main()
