"""Offline independent cleanup contracts; no live model or storage acceptance."""

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from doc_loaders.extraction_identity import ExtractionIdentity, ExtractionResult, text_digest
from document_structure.cleanup_input import CleanupInput, CleanupViewPart
from document_structure.cleanup_operation import (
    CleanupCommitCommand, CleanupCommitResult, CleanupOperationBinding,
    CleanupOperationState, CleanupRepository, StagedCleanup,
)
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, CanonicalOcrSpan, CleanupOutcome
from document_structure.ocr_cleanup_validation import CleanupItemRejection, ValidatedDeletion
from document_structure.source_mapped_text import PageRange
from scripts.test_ocr_cleanup_contract import evidence, item, proposal
from scripts.test_ocr_cleanup_validation import fixture


def snapshot(*, mode="ocr", revision=0, document_update=None, **changes):
    source, _, document = fixture(mode=mode)
    provenance = {**document.parse_provenance, "extraction": source.record(),
                  "parser_mode": "common", "reading_content": {"revision": revision}}
    document = replace(document, parse_provenance=provenance)
    payload = document.to_dict()
    if document_update:
        document_update(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return CleanupInput(**{
        "namespace": "test", "document_id": document.document_id, "source": source,
        "structure_version": 1, "hierarchy_json": encoded,
        "hierarchy_fingerprint": text_digest(encoded), "reading_revision": revision,
        "evidence": None, "parts": (CleanupViewPart(char_start=0, char_end=source.text_length,
                                                      text=source.canonical_text),), **changes})


def binding(value, **changes):
    return CleanupOperationBinding(**{
        "operation_id": "cleanup-1", "namespace": value.namespace, "document_id": value.document_id,
        "input_fingerprint": value.fingerprint(), "expected_reading_revision": value.reading_revision,
        "prompt_version": "cleanup-test-v1", "policy_version": "cleanup-test-v1",
        "capacity_fingerprint": "a" * 64, "consent_scope": "standalone_ocr_cleanup_v1", **changes})


def outcome(*, changed=False, revision=0, **changes):
    return CleanupOutcome(**{
        "status": "completed" if changed else "no_changes", "proposed_count": int(changed),
        "applied_count": int(changed), "skipped_count": 0, "reason": None,
        "reading_content_changed": changed, "reading_revision": revision + int(changed), **changes})


def stage(value, *, empty=False, **changes):
    return StagedCleanup(**{
        "binding": binding(value), "proposal": proposal([] if empty else [item(quote="@@??")]),
        "deletions": () if empty else (ValidatedDeletion(0, 0, 4, "c1", "s0", "@@??", "ocr_garbage", ()),),
        "rejections": (), **changes})


class CleanupInputTests(unittest.TestCase):
    def test_common_without_units_or_splitter_is_eligible(self):
        with patch("document_structure.ocr_cleanup_planner.plan_ocr_cleanup",
                   side_effect=AssertionError("must not call planner")):
            value = snapshot()
            self.assertEqual(value.eligibility().status, "eligible")
            self.assertEqual(value.document().chapters[0].sections[0].task_units, [])
            self.assertEqual(value.document().parse_provenance["parser_mode"], "common")

    def test_snapshot_is_deeply_immutable_and_detached(self):
        value = snapshot()
        before = value.fingerprint()
        value.document().chapters.clear()
        self.assertEqual(value.fingerprint(), before)
        self.assertTrue(value.document().chapters)
        with self.assertRaises(ValidationError):
            value.parts[0].text = "changed"
        with self.assertRaises(ValidationError):
            value.reading_revision = 2

    def test_json_roundtrip(self):
        value = snapshot()
        self.assertEqual(CleanupInput.model_validate_json(value.model_dump_json()), value)

    def test_unknown_fields_and_coercions_denied(self):
        for changes in ({"accepted": True}, {"parser_mode": "llm_enhanced"},
                        {"structure_version": True}, {"reading_revision": "0"},
                        {"parts": []}, {"structure_version": 0}, {"schema_version": "other"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                snapshot(**changes)

    def test_bad_hierarchy_source_and_revision_denied(self):
        mutations = [lambda p: p.update(chapters=[]), lambda p: p.update(document_id="other"),
                     lambda p: p.update(raw_text="different"), lambda p: p.update(sections=[]),
                     lambda p: p["parse_provenance"].pop("extraction"),
                     lambda p: p["parse_provenance"]["extraction"].update(text_sha256="a" * 64),
                     lambda p: p["parse_provenance"].update(reading_content={"revision": 1}),
                     lambda p: p["parse_provenance"].update(reading_content={"revision": False}),
                     lambda p: p["chapters"][0]["sections"][0].update(char_start=2)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "cleanup_snapshot_invalid"):
                snapshot(document_update=mutate)

    def test_bad_snapshot_digest_denied(self):
        with self.assertRaisesRegex(ValueError, "cleanup_snapshot_invalid"):
            snapshot(hierarchy_fingerprint="0" * 64)

    def test_auto_requires_current_actual_ocr_evidence(self):
        value = snapshot(mode="auto")
        self.assertEqual(value.eligibility().reason, "ocr_evidence_missing")
        current = snapshot(mode="auto", evidence=evidence(value.source))
        self.assertEqual(current.eligibility().status, "eligible")
        with self.assertRaisesRegex(ValueError, "ocr_evidence_mismatch"):
            snapshot(mode="auto", evidence=evidence(snapshot().source))
        with self.assertRaisesRegex(ValueError, "ocr_evidence_mismatch"):
            snapshot(evidence=evidence(value.source))

    def test_native_and_unknown_never_gain_eligibility(self):
        for mode in ("fast", "native", "unknown"):
            value = snapshot(mode=mode)
            self.assertNotEqual(value.eligibility().status, "eligible")
            with self.assertRaises(ValueError):
                CleanupCommitCommand(snapshot=value, staged=stage(value, empty=True))

    def test_disjoint_view_requires_explicit_deleted_gap(self):
        value = snapshot()
        part = CleanupViewPart(char_start=4, char_end=value.source.text_length,
                               text=value.source.canonical_text[4:])
        with self.assertRaisesRegex(ValueError, "cleanup_view_coverage_invalid"):
            snapshot(parts=(part,))
        value = snapshot(parts=(part,), prior_deletions=(CanonicalOcrSpan(char_start=0, char_end=4),))
        self.assertNotIn("@@??", "".join(p.text for p in value.parts))
        with self.assertRaisesRegex(ValueError, "cleanup_deletion_outside_view"):
            CleanupCommitCommand(snapshot=value, staged=stage(value))

    def test_manual_context_must_be_protected(self):
        value = snapshot()
        part = CleanupViewPart(char_start=0, char_end=value.source.text_length,
                               text="User replacement", manual=True)
        with self.assertRaisesRegex(ValueError, "cleanup_manual_context_unprotected"):
            snapshot(parts=(part,))
        protected = (CanonicalOcrSpan(char_start=0, char_end=value.source.text_length),)
        current = snapshot(parts=(part,), protected_ranges=protected)
        with self.assertRaisesRegex(ValueError, "cleanup_deletion_outside_view"):
            CleanupCommitCommand(snapshot=current, staged=stage(current))
        with self.assertRaisesRegex(ValueError, "cleanup_view_text_mismatch"):
            snapshot(parts=(CleanupViewPart(char_start=0, char_end=value.source.text_length, text="invented"),))

    def test_pages_allow_empty_pages_but_not_unmapped_text(self):
        value = snapshot()
        snapshot(pages=(PageRange(0, 0, 0), PageRange(1, 0, value.source.text_length)))
        for pages in ((PageRange(0, 1, value.source.text_length),),
                      (PageRange(0, 0, value.source.text_length - 2),),
                      (PageRange(False, 0, value.source.text_length),)):
            with self.subTest(pages=pages), self.assertRaisesRegex(ValueError, "cleanup_pages_invalid"):
                snapshot(pages=pages)


class CleanupCommandTests(unittest.TestCase):
    def test_consent_binds_all_snapshot_data(self):
        value = snapshot()
        consent = binding(value)
        for current in (snapshot(revision=1), snapshot(structure_version=2),
                        snapshot(namespace="other"), snapshot(evidence=evidence(value.source)),
                        snapshot(protected_ranges=(CanonicalOcrSpan(char_start=0, char_end=4),))):
            self.assertFalse(consent.matches(current))
        self.assertTrue(consent.matches(value))

    def test_standalone_does_not_accept_structural_discard_permission(self):
        for changes in ({"consent_scope": "ocr_cleanup_reading_overlay_v1"},
                        {"discard_interactions": True}, {"capacity_fingerprint": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                binding(snapshot(), **changes)

    def test_command_and_stage_roundtrip_and_mismatch(self):
        value = snapshot()
        command = CleanupCommitCommand(snapshot=value, staged=stage(value))
        self.assertEqual(CleanupCommitCommand.model_validate_json(command.model_dump_json()), command)
        with self.assertRaisesRegex(ValueError, "cleanup_input_binding_mismatch"):
            CleanupCommitCommand(snapshot=snapshot(revision=1), staged=stage(value))

    def test_stage_accounts_for_every_proposal_once(self):
        value = snapshot()
        for changes in ({"deletions": ()}, {"rejections": (CleanupItemRejection(0, "protected"),)},
                        {"deletions": (replace(stage(value).deletions[0], quote="invented"),)},
                        {"deletions": (replace(stage(value).deletions[0], char_start=True),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                stage(value, **changes)
        stage(value, deletions=(), rejections=(CleanupItemRejection(0, "protected_content"),))

    def test_protected_canonical_text_cannot_be_deleted(self):
        value = snapshot(protected_ranges=(CanonicalOcrSpan(char_start=0, char_end=4),))
        with self.assertRaisesRegex(ValueError, "cleanup_deletion_protected"):
            CleanupCommitCommand(snapshot=value, staged=stage(value))

    def test_noop_does_not_advance_revision(self):
        value = snapshot(revision=3)
        CleanupCommitCommand(snapshot=value, staged=stage(value, empty=True))
        for replayed in (False, True):
            result = CleanupCommitResult(binding=binding(value), outcome=outcome(revision=3),
                                         replayed=replayed, reading_layout_required=False)
            self.assertEqual(result.outcome.reading_revision, 3)
        with self.assertRaisesRegex(ValueError, "cleanup_commit_revision_mismatch"):
            CleanupCommitResult(binding=binding(value), outcome=outcome(revision=4),
                                reading_layout_required=False)

    def test_changed_commit_advances_once_and_replay_preserves_receipt(self):
        value = snapshot(revision=3)
        for replayed in (False, True):
            result = CleanupCommitResult(binding=binding(value), outcome=outcome(changed=True, revision=3),
                                         replayed=replayed, reading_layout_required=True)
            self.assertEqual(result.outcome.reading_revision, 4)
        self.assertNotEqual(binding(value), binding(value, operation_id="new-consented-attempt"))

    def test_repository_port_is_narrow_not_an_executor_or_job_framework(self):
        public = {name for name in CleanupRepository.__dict__ if not name.startswith("_")}
        self.assertEqual(public, {"commit_cleanup", "lookup_cleanup"})


class CleanupOperationStateTests(unittest.TestCase):
    def state(self, **changes):
        return CleanupOperationState(**{
            "binding": binding(snapshot()), "phase": "planning", "commit": "not_started",
            "outcome": None, "retry": "none", **changes})

    def test_phases_do_not_claim_early_success(self):
        self.state()
        self.state(phase="staging")
        self.state(phase="committing", commit="unknown")
        for phase in ("planning", "staging", "committing", "unknown"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                self.state(phase=phase, outcome=outcome(changed=True))

    def test_unknown_requires_lookup_not_regeneration(self):
        self.state(phase="unknown", commit="unknown", retry="reconcile")
        for retry in ("none", "regenerate_cleanup", "resume_layout"):
            with self.subTest(retry=retry), self.assertRaises(ValueError):
                self.state(phase="unknown", commit="unknown", retry=retry)

    def test_layout_failure_preserves_committed_cleanup_and_resumes_only_layout(self):
        self.state(phase="layout", commit="committed", outcome=outcome(changed=True), retry="resume_layout")
        with self.assertRaises(ValueError):
            self.state(phase="layout", commit="committed", outcome=outcome(changed=True), retry="regenerate_cleanup")

    def test_provider_failure_allows_new_consented_attempt_but_capacity_does_not(self):
        self.state(phase="finished", commit="not_committed", retry="regenerate_cleanup",
                   outcome=outcome(status="failed", proposed_count=None, reason="provider_failed"))
        with self.assertRaisesRegex(ValueError, "cleanup_retry_not_available"):
            self.state(phase="finished", commit="not_committed", retry="regenerate_cleanup",
                       outcome=outcome(status="skipped", proposed_count=None, reason="capacity_exceeded"))

    def test_completed_superseded_and_partial_states(self):
        self.state(phase="finished", commit="committed", outcome=outcome(changed=True))
        self.state(phase="superseded", commit="committed", outcome=outcome(changed=True))
        self.state(phase="finished", commit="committed", retry="regenerate_cleanup",
                   outcome=outcome(changed=True, status="partial", proposed_count=2,
                                   skipped_count=1, reason="some_items_rejected"))
        for changes in ({"phase": "superseded", "commit": "unknown", "retry": "regenerate_cleanup"},
                        {"phase": "finished", "commit": "not_committed", "outcome": outcome(changed=True)},
                        {"phase": "finished", "commit": "committed"},
                        {"phase": "finished", "commit": "committed", "outcome": outcome(revision=2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.state(**changes)

    def test_contracts_do_not_import_upper_layers_or_execute_external_work(self):
        root = Path(__file__).resolve().parents[1] / "document_structure"
        for name in ("cleanup_input.py", "cleanup_operation.py"):
            tree = ast.parse((root / name).read_text())
            imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            self.assertFalse(any(module and module.split(".")[0] in {
                "app", "main", "api_schemas", "db", "document_preparation", "llm"} for module in imports))
            calls = [n.func.id for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertFalse(set(calls) & {"open", "plan_ocr_cleanup", "prepare_enhancement_cleanup"})


class PrivateCleanupInputTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"),
                         "opt-in backed-up Common/OCR contract fixtures")
    def test_real_common_ocr_backups_read_only(self):
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                directory = root / name
                paths = [directory / file for file in ("book.pdf", "document.json", "manifest.json")]
                before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                manifest = json.loads(paths[2].read_text())
                self.assertEqual(before[paths[0]], manifest["source_sha256"])
                payload = json.loads(paths[1].read_text())
                provenance = payload["parse_provenance"]
                self.assertEqual(provenance["effective_parser_mode"], "common")
                source = ExtractionResult(ExtractionIdentity(**provenance["extraction"]["identity"]),
                                          payload["raw_text"])
                self.assertEqual(source.identity.source_sha256, before[paths[0]])
                self.assertEqual(source.record(), provenance["extraction"])
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                current = CleanupInput(namespace="contract-test", document_id=payload["document_id"],
                    source=source, structure_version=1, hierarchy_json=encoded,
                    hierarchy_fingerprint=text_digest(encoded), reading_revision=0,
                    evidence=ActualOcrEvidence.model_validate_json(json.dumps(provenance["actual_ocr_evidence"])),
                    pages=tuple(PageRange(p["page_index"], p["char_start"], p["char_end"])
                                for p in provenance["source_page_spans"]),
                    parts=(CleanupViewPart(char_start=0, char_end=source.text_length, text=source.canonical_text),))
                self.assertEqual(current.eligibility().status, "eligible")
                self.assertEqual(CleanupInput.model_validate_json(current.model_dump_json()), current)
                self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


if __name__ == "__main__":
    unittest.main()
