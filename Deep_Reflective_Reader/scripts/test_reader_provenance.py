"""READER-01 bounded provenance contract; no providers or real documents."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import main
from app.section_task_coordinator import SectionTaskCoordinator
from section_tasks.document_task_layout import DocumentTaskLayout
from test_task_layout_chapters_response import (
    _build_layout_ready_document, _build_coordinator,
    StructuredDocumentArtifactRepository, StructuredDocumentStore,
)


class ReaderProvenanceTests(unittest.TestCase):
    def test_section_origins_match_exact_accepted_ranges_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = StructuredDocumentArtifactRepository(store=StructuredDocumentStore(), base_dir=directory)
            document = _build_layout_ready_document(with_chapters=True)
            document.document_task_artifacts.metadata["task_layout"]["resolver_version"] = SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION
            section = document.chapters[0].sections[0]
            document = replace(document, parse_provenance={"boundary_origins": [
                {"char_start": section.char_start, "char_end": section.char_end, "title_origin": "generated"},
                {"char_start": -1, "char_end": -2, "title_origin": "source"},
            ]})
            repository.save_document(document, doc_name="layout-doc")
            before = (Path(directory) / "layout-doc.structured.json").read_bytes()
            coordinator = _build_coordinator(repository)
            with patch.object(main, "section_task_coordinator", coordinator):
                response = TestClient(main.app).post("/documents/task-layout", json={
                    "doc_name": "layout-doc", "task_unit_split_mode": "semantic_safe", "semantic_top_k_candidates": 3,
                })
            self.assertEqual(response.status_code, 200, response.text)
            chapters = response.json()["chapters"]
            self.assertEqual(chapters[0]["sections"][0]["title_origin"], "generated")
            self.assertIsNone(chapters[1]["sections"][0]["title_origin"])
            self.assertEqual(before, (Path(directory) / "layout-doc.structured.json").read_bytes())

    def test_legacy_unknown_and_invalid_fields(self):
        project = SectionTaskCoordinator._build_parse_provenance
        self.assertIsNone(project({}))
        result = project({"source": "legacy", "common_parse_policy": {
            "toc_hint": "invented", "implicit_single_body": "true", "toc_observation": "raw secret",
        }}).to_dict()
        for key in ("toc_hint", "implicit_single_body", "toc_observation", "declaration_unconfirmed"):
            self.assertIsNone(result[key])

    def test_route_projects_whitelisted_metadata_without_mutation(self):
        provenance = {"effective_parser_mode": "common", "requested_parser_mode": "common",
            "raw_text": "private", "boundary_origins": [{"char_start": 0}],
            "common_parse_policy": {"toc_hint": "present", "toc_observation": "not_detected",
                "implicit_single_body": True, "declaration_unconfirmed": True,
                "enhancement_intent": "restore_source", "prompt": "private"}}
        before = deepcopy(provenance)
        dto = SectionTaskCoordinator._build_parse_provenance(provenance)
        layout = DocumentTaskLayout(document_id="Book", title="Book", language="en", chapters=[], parse_provenance=dto)
        coordinator = Mock()
        coordinator.get_document_task_layout.return_value = layout
        with patch.object(main, "section_task_coordinator", coordinator):
            response = TestClient(main.app).post("/documents/task-layout", json={"doc_name": "Book"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["parse_provenance"]
        self.assertTrue(result["implicit_single_body"])
        self.assertTrue(result["declaration_unconfirmed"])
        self.assertEqual(result["toc_hint"], "present")
        self.assertEqual(result["toc_observation"], "not_detected")
        self.assertNotIn("private", response.text)
        self.assertNotIn("boundary_origins", result)
        self.assertEqual(provenance, before)

    def test_enhanced_observation_and_unknown_remain_distinct(self):
        for observation, expected in [("not_found", True), ("not_checked", None)]:
            dto = SectionTaskCoordinator._build_parse_provenance({
                "toc_observation": observation, "enhancement_intent": "generate_reading_outline",
                "common_parse_policy": {"toc_hint": "present"}})
            self.assertEqual(dto.declaration_unconfirmed, expected)
            self.assertEqual(dto.enhancement_intent, "generate_reading_outline")


if __name__ == "__main__":
    unittest.main()
