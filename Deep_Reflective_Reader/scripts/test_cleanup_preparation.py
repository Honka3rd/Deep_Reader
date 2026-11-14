"""Cleanup handoff and explicit layout against isolated accepted sources."""

from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

from doc_loaders.extraction_identity import ExtractionResult
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
from document_preparation.cleanup_handoff import accepted_cleanup_context, prepare_enhancement_cleanup, extraction_cleanup_metadata
from document_preparation.reading_layout_handoff import resolve_reading_layout
from document_structure.ocr_cleanup_contract import cleanup_eligibility
from document_structure.ocr_cleanup_overlay import unit_ranges
from document_structure.ocr_cleanup_planner import CLEANUP_PROMPT_VERSION
from document_structure.reading_content import ReadingContentConflict, project_document
from section_tasks.task_unit_resolver import TaskUnitResolver
from scripts.test_ocr_cleanup_contract import source, evidence
from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests
from scripts.test_ocr_cleanup_planner import Provider
from scripts.test_task_layout_readiness import ReadinessTests
from scripts.test_ocr_cleanup_overlay_postgres import PostgresCleanupOverlayTests
from doc_loaders.source_session import SourceArtifact, SourceTextSpan


class CleanupPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CleanupOverlayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=100)

    def accepted(self):
        f = self.fixture
        document, version, _ = f.snapshot()
        return resolve_accepted_source_snapshot(document=document, namespace=f.namespace,
            expected_document_id="book", structure_version=version, source_path=f.path, accepted=True)

    def planned(self, provider=None, kind="structure_changed"):
        candidate, application = self.fixture.prepare(kind)
        candidate = replace(candidate, source=self.accepted())
        provider = provider or Provider()
        provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
            {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"}]})
        result = prepare_enhancement_cleanup(candidate=candidate, binding=application.plan.binding.model_copy(
            update={"prompt_version": CLEANUP_PROMPT_VERSION}),
                                             provider=provider)
        return candidate, result, provider

    def layout(self, document):
        calls = []
        def resolver(**kwargs):
            calls.append(kwargs["document"])
            return self.resolver.resolve_for_task_layout(**kwargs)
        units = resolve_reading_layout(document=document, resolver=resolver, split_mode="progressive")
        updated = replace(document, chapters=[replace(c, sections=[replace(s,
            task_units=[u for u in units if u.source_section_ids == [s.section_id]])
            for s in c.sections]) for c in document.chapters])
        return updated, calls

    def test_snapshot_carries_pages_and_no_loader_profile_calls(self):
        snapshot = self.accepted()
        self.assertEqual(snapshot.cleanup_context.pages[0].end, snapshot.text_length)
        self.assertIsNone(snapshot.cleanup_context.evidence)
        self.assertEqual(snapshot.canonical_text, self.fixture.text)

    def test_plan_uses_in_memory_candidate_and_never_saves_or_reloads(self):
        candidate, application = self.fixture.prepare()
        candidate = replace(candidate, source=self.accepted())
        before = candidate.document.to_dict()
        with patch.object(self.fixture.repo, "load_document", side_effect=AssertionError("reload")), \
                patch.object(self.fixture.repo, "save_document", side_effect=AssertionError("save")):
            planned = prepare_enhancement_cleanup(candidate=candidate,
                binding=application.plan.binding.model_copy(update={"prompt_version": CLEANUP_PROMPT_VERSION}), provider=Provider())
        self.assertIsNotNone(planned.application)
        self.assertIsNone(planned.outcome)
        self.assertEqual(before, candidate.document.to_dict())

    def test_preunit_cleanup_empty_section_mapping_and_undo(self):
        candidate, planned, provider = self.planned()
        self.assertEqual(len(provider.calls), 1)
        committed = self.fixture.commit(candidate, planned.application)
        document, calls = self.layout(committed.document)
        self.assertTrue(calls)
        self.assertTrue(all("@@??" not in s.content for d in calls for c in d.chapters for s in c.sections))
        self.assertEqual("".join(u.content for c in document.chapters for s in c.sections for u in s.task_units), self.fixture.text)
        for chapter in document.chapters:
            for section in chapter.sections:
                self.assertTrue(unit_ranges(section))
        self.fixture.repo.save_document(document, "book")
        effective = project_document(self.fixture.repo.load_document("book"))
        self.assertEqual(effective.chapters[0].sections[0].task_units[0].content, "")
        self.assertNotIn("@@??", self.fixture.content())
        self.fixture.change("undo")
        self.assertIn("@@??", self.fixture.content())

    def test_cleanup_only_reuses_unit_ids_without_resolver(self):
        old_id = self.fixture.repo.load_document("book").chapters[0].sections[0].task_units[0].unit_id
        candidate, planned, _ = self.planned(kind="unchanged")
        committed = self.fixture.commit(candidate, planned.application)
        document, calls = self.layout(committed.document)
        self.assertEqual(calls, [])
        self.assertEqual(document.chapters[0].sections[0].task_units[0].unit_id, old_id)
        self.assertNotIn("@@??", project_document(document).chapters[0].sections[0].content)

    def test_failed_cleanup_keeps_candidate_usable_for_layout(self):
        provider = Provider()
        provider.error = TimeoutError("private provider failure")
        candidate, planned, _ = self.planned(provider)
        self.assertIsNone(planned.application)
        self.assertEqual(planned.outcome.reason, "provider_failed")
        document, calls = self.layout(candidate.document)
        self.assertTrue(calls)
        self.assertIn("@@??", document.chapters[0].sections[0].task_units[0].content)

    def test_invalid_structure_never_calls_cleanup(self):
        candidate, application = self.fixture.prepare()
        candidate = replace(candidate, document=replace(candidate.document, raw_text="bad"))
        provider = Provider()
        with self.assertRaises(ValueError):
            prepare_enhancement_cleanup(candidate=candidate, binding=application.plan.binding, provider=provider)
        self.assertEqual(provider.calls, [])

    def test_auto_missing_evidence_never_calls_provider(self):
        candidate, application = self.fixture.prepare()
        identity = replace(candidate.source.extraction_identity, mode="auto")
        candidate = replace(candidate, source=replace(candidate.source, extraction_identity=identity))
        bound = application.plan.binding.model_copy(update={"extraction_identity": identity,
                                                            "prompt_version": CLEANUP_PROMPT_VERSION})
        provider = Provider()
        result = prepare_enhancement_cleanup(candidate=candidate, binding=bound, provider=provider)
        self.assertIsNone(result.application)
        self.assertEqual(result.outcome.reason, "ocr_evidence_missing")
        self.assertEqual(provider.calls, [])

    def test_whitespace_normalizing_resolver_retains_exact_canonical_units(self):
        candidate, planned, _ = self.planned()
        document = self.fixture.commit(candidate, planned.application).document
        def normalizing(**kwargs):
            return [replace(u, content=" ".join(u.content.split()))
                    for u in self.resolver.resolve_for_task_layout(**kwargs)]
        units = resolve_reading_layout(document=document, resolver=normalizing, split_mode="progressive")
        self.assertEqual("".join(u.content for u in units), self.fixture.text)

    def test_manual_source_replacement_survives_new_units(self):
        block = self.fixture.service.read("book")["blocks"][1]
        self.fixture.change("edit", block_id=block["block_id"], text="User-authoritative replacement.")
        candidate, planned, _ = self.planned()
        document, _ = self.layout(self.fixture.commit(candidate, planned.application).document)
        projected = project_document(document)
        effective = " ".join(s.content for c in projected.chapters for s in c.sections)
        self.assertIn("User-authoritative replacement.", effective)
        self.assertNotIn("@@??", effective)

    def test_source_changed_rejects_layout_without_resolving(self):
        candidate, planned, _ = self.planned()
        document = self.fixture.commit(candidate, planned.application).document
        resolver = Mock(side_effect=AssertionError("resolve"))
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_source_changed"):
            resolve_reading_layout(document=replace(document, raw_text="changed"), resolver=resolver)

    def test_resolver_rewriting_text_cannot_be_persisted(self):
        candidate, planned, _ = self.planned()
        document = self.fixture.commit(candidate, planned.application).document
        def rewriting(**kwargs):
            return [replace(u, content="invented") for u in self.resolver.resolve_for_task_layout(**kwargs)]
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_layout_mapping_unavailable"):
            resolve_reading_layout(document=document, resolver=rewriting, split_mode="progressive")

    def test_actual_explicit_layout_pipeline_and_repeated_prepare(self):
        candidate, planned, _ = self.planned()
        self.fixture.commit(candidate, planned.application)
        readiness = ReadinessTests()
        readiness.setUp()
        self.addCleanup(readiness.doCleanups)
        coordinator = readiness.coordinator
        coordinator.document_artifact_repository = self.fixture.repo
        coordinator.task_unit_resolver = self.resolver
        options = {"doc_name": "book", "task_unit_split_mode": "progressive"}
        coordinator.ensure_document_task_layout(**options)
        document = self.fixture.repo.load_document("book")
        for chapter in document.chapters:
            for section in chapter.sections:
                self.assertTrue(unit_ranges(section))
        self.assertNotIn("@@??", self.fixture.content())
        before = document.to_dict()
        with patch.object(self.fixture.repo, "update_task_layout", side_effect=AssertionError("rewrote")):
            coordinator.ensure_document_task_layout(**options)
            coordinator.get_document_task_layout(**options)
        self.assertEqual(before, self.fixture.repo.load_document("book").to_dict())
        readiness.pipeline.prepare_and_load.assert_not_called()

    def test_stale_layout_write_rejected_after_reading_edit(self):
        document, _, _ = self.fixture.snapshot()
        self.fixture.change("delete", block_id=self.fixture.service.read("book")["blocks"][0]["block_id"])
        with self.assertRaisesRegex(ReadingContentConflict, "reading_revision_changed"):
            from scripts.artifact_generation_fixtures import bound_layout
            self.fixture.repo.update_task_layout("book", {}, bound_layout(document, {"reading_revision": 0}))
        changed = replace(document, chapters=[replace(c, sections=[replace(s, task_units=[])
            for s in c.sections]) for c in document.chapters])
        with self.assertRaisesRegex(ReadingContentConflict, "reading_revision_changed"):
            self.fixture.repo.save_document(changed, "book")


class PostgresCleanupPreparationTests(CleanupPreparationTests):
    setUpClass = classmethod(PostgresCleanupOverlayTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresCleanupOverlayTests.drop_schema.__func__)

    def setUp(self):
        self.fixture = PostgresCleanupOverlayTests()
        self.fixture.scoped = self.scoped
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=100)

    def test_failed_cleanup_commits_source_evidence_without_another_extraction(self):
        provider = Provider()
        provider.error = TimeoutError()
        candidate, planned, _ = self.planned(provider)
        current = self.fixture.repo.load_enhancement_snapshot("book")
        result = self.fixture.repo.commit_enhancement("book", candidate=candidate,
            interaction_revision=current.interaction_revision, source_path=self.fixture.path,
            cleanup=planned.application)
        self.assertIn("source_page_spans", result.document.parse_provenance)
        self.assertIn("source_review", result.document.parse_provenance)
        self.assertIsNone(result.cleanup)
        document, calls = self.layout(result.document)
        self.assertTrue(calls)
        self.assertIn("@@??", document.chapters[0].sections[0].content)


class CleanupEvidenceTests(unittest.TestCase):
    def test_source_capability_persists_exact_identity_and_coordinates(self):
        value = source("auto")
        spans = (SourceTextSpan(0, 3, 0),)
        record = extraction_cleanup_metadata(SourceArtifact(value, ocr_spans=spans))
        context = accepted_cleanup_context(value, record)
        self.assertEqual(context.evidence.spans[0].char_end, 3)
        self.assertEqual(context.evidence.extraction_identity, value.identity)
        self.assertEqual(context.evidence.text_sha256, value.text_sha256)
        self.assertEqual(extraction_cleanup_metadata(SourceArtifact(value)), {})
        native = accepted_cleanup_context(value, extraction_cleanup_metadata(SourceArtifact(value, ocr_spans=())))
        self.assertEqual(cleanup_eligibility(value, native.evidence).status, "not_applicable")

    def test_pdf_adapter_owns_actual_auto_recovery_selection(self):
        from pathlib import Path
        import tempfile
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        from doc_loaders.pdf_extraction_contract import PdfExtractionMode
        from doc_loaders.pdf_page_evidence import PdfPageTextBoundary, PdfPageBoundaryEvidenceResult
        from doc_loaders.pdf_extraction_result import PdfExtractionResult
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "book.pdf").write_bytes(b"fixture bytes; mocked extractor")
            loader = PdfDocumentLoader(base_dir=Path(directory))
            for mode, provenance, expected in (("fast", {}, []), ("ocr", {}, [0, 1]),
                    ("auto", {}, []), ("auto", {"recovery_page_indexes": [1]}, [1]),
                    ("auto", {"unknown": True}, None)):
                with self.subTest(mode=mode, provenance=provenance):
                    session = loader.open_source("book", mode=mode)
                    boundaries = [PdfPageTextBoundary(page_index=i, char_start=i*4, char_end=i*4+3, text=t)
                                  for i, t in enumerate(("abc", "def"))]
                    result = PdfExtractionResult(text="abc\ndef", pages=("abc", "def"),
                        boundary_evidence=PdfPageBoundaryEvidenceResult("book", "book.pdf", session.identity.source_sha256, 2, boundaries),
                        effective_mode=PdfExtractionMode(mode), extraction_version=session.identity.extraction_version,
                        empty_page_indexes=(), warnings=(), layout_provider=None, outline_provider=None, provenance=provenance)
                    with patch.object(loader, "extract", return_value=result) as extract:
                        artifact = session.extract()
                        self.assertIs(artifact, session.extract())
                        extract.assert_called_once()
                    actual = None if artifact.ocr_spans is None else [p.page_index for p in artifact.ocr_spans]
                    self.assertEqual(actual, expected)

    def test_auto_evidence_json_roundtrip_and_missing(self):
        value = source("auto")
        context = accepted_cleanup_context(value, {"actual_ocr_evidence": evidence(value).model_dump(mode="json")})
        self.assertEqual(cleanup_eligibility(value, context.evidence).status, "eligible")
        self.assertEqual(cleanup_eligibility(value, accepted_cleanup_context(value, {}).evidence).reason,
                         "ocr_evidence_missing")

    def test_malformed_and_stale_evidence_never_qualifies(self):
        value = source("auto")
        for raw in ({"bad": True}, evidence(source("ocr")).model_dump(mode="json")):
            with self.subTest(raw=raw):
                context = accepted_cleanup_context(value, {"actual_ocr_evidence": raw})
                self.assertEqual(cleanup_eligibility(value, context.evidence).status, "skipped")

    def test_generic_third_format_capability_not_suffix_dispatch(self):
        value = source("ocr")
        value = ExtractionResult(replace(value.identity, source_format="epub", extractor_id="fixture"), value.canonical_text)
        self.assertEqual(cleanup_eligibility(value, accepted_cleanup_context(value, {}).evidence).status, "eligible")
        native = ExtractionResult(replace(value.identity, mode="native"), value.canonical_text)
        self.assertEqual(cleanup_eligibility(native).status, "not_applicable")

    def test_bad_pages_are_optional_and_never_accepted_as_coordinates(self):
        value = source()
        for spans in (None, [{"page_index": 0, "char_start": 1, "char_end": value.text_length}],
                      [{"page_index": True, "char_start": 0, "char_end": value.text_length}]):
            self.assertEqual(accepted_cleanup_context(value, {"source_page_spans": spans}).pages, ())


class CleanupLayoutCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ReadinessTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_reading_revision_requires_explicit_prepare_and_then_reuses(self):
        f = self.fixture
        f.prepare()
        document = f.repository.load_document("cache-doc")
        f.repository.mutate_reading_content("cache-doc", lambda current: (replace(current,
            parse_provenance={**current.parse_provenance, "reading_content": {"revision": 1,
                "cleanup_receipts": {"fixture": {}}}}), None))
        f.read_without_side_effects(reason="reading_revision_mismatch")
        f.coordinator.ensure_document_task_layout(**f.options)
        f.read_without_side_effects()
        with patch.object(f.repository, "update_task_layout", side_effect=AssertionError("unnecessary write")):
            f.coordinator.ensure_document_task_layout(**f.options)

    def test_legacy_manual_review_revision_does_not_break_ready_layout(self):
        f = self.fixture
        f.prepare()
        f.repository.mutate_reading_content("cache-doc", lambda current: (replace(current,
            parse_provenance={**current.parse_provenance, "reading_content": {"revision": 1}}), None))
        f.read_without_side_effects()


if __name__ == "__main__":
    unittest.main()
