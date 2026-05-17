#!/usr/bin/env python3
"""Pure hierarchy JSON cleanup smoke tests."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import replace
from pathlib import Path

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.structured_hierarchy_builder import DocumentHierarchyBuilder
from shared.task_artifacts import DocumentTaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_hierarchy_document() -> StructuredDocument:
    section = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="第一章",
        level=1,
        content="章節內容",
        char_start=0,
        char_end=4,
        section_role=None,
        parent_chapter_id="chapter-1",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-local",
                title="u1",
                container_title=None,
                content="章節內容",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id=None,
            )
        ],
    )
    return StructuredDocument(
        document_id="cleanup-doc",
        title="Cleanup Doc",
        source_path=None,
        language="zh",
        raw_text="第一章\n章節內容",
        sections=[],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-1",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[section],
            )
        ],
        structure_nodes=[],
        document_task_artifacts=DocumentTaskArtifacts(),
    )


def _build_legacy_sections_only_document() -> StructuredDocument:
    return StructuredDocument(
        document_id="legacy-only-doc",
        title="Legacy Only",
        source_path=None,
        language="zh",
        raw_text="前言\n文本\n\n第一章\n正文",
        sections=[
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="前言",
                level=1,
                content="前言\n文本",
                char_start=0,
                char_end=5,
                section_role=SectionRole.FRONT_MATTER,
            ),
            StructuredSection(
                section_id="section-1",
                section_index=1,
                title="第一章",
                level=1,
                content="第一章\n正文",
                char_start=6,
                char_end=11,
                section_role=SectionRole.MAIN_BODY,
                task_units=[
                    TaskUnit(
                        unit_id="legacy-unit-0",
                        title="legacy-u0",
                        container_title=None,
                        content="第一章\n正文",
                        source_section_ids=["section-1"],
                        is_fallback_generated=False,
                        parent_section_id=None,
                    )
                ],
            ),
        ],
        chapters=[],
    )


def _task_unit_object_paths(payload: object) -> list[str]:
    paths: list[str] = []

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            is_task_unit_object = (
                "unit_id" in node
                and "source_section_ids" in node
                and "is_fallback_generated" in node
            )
            if is_task_unit_object:
                paths.append(path)
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else key
                _walk(value, child_path)
            return

        if isinstance(node, list):
            for index, value in enumerate(node):
                child_path = f"{path}[{index}]"
                _walk(value, child_path)

    _walk(payload, "")
    return paths


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_new_json_has_no_legacy_mirror() -> None:
    document = _build_hierarchy_document()
    serialized = json.loads(document.to_json())
    _assert("sections" not in serialized, "new JSON should not persist root sections mirror")
    _assert(
        "structure_nodes" not in serialized,
        "new JSON should not persist root structure_nodes mirror",
    )
    _assert(serialized.get("chapters"), "new JSON should contain chapters")


def test_to_dict_omits_root_sections_even_when_chapters_empty_by_default() -> None:
    legacy_only_document = _build_legacy_sections_only_document()
    serialized = legacy_only_document.to_dict()
    _assert(
        "sections" not in serialized,
        "to_dict default should not emit root sections even when chapters are empty",
    )


def test_to_dict_includes_root_sections_when_explicitly_enabled() -> None:
    legacy_only_document = _build_legacy_sections_only_document()
    serialized = legacy_only_document.to_dict(include_legacy_sections=True)
    _assert(
        "sections" in serialized,
        "to_dict should emit root sections when include_legacy_sections=True",
    )
    _assert(
        len(serialized["sections"]) == 2,
        "explicit legacy serialization should keep legacy sections payload",
    )


def test_from_dict_fails_for_legacy_sections_only_payload() -> None:
    legacy_payload = {
        "document_id": "legacy-sections-payload-doc",
        "title": "Legacy Sections Payload",
        "source_path": None,
        "language": "zh",
        "raw_text": "前言\\n文本",
        "sections": [
            {
                "section_id": "section-legacy-0",
                "section_index": 0,
                "title": "前言",
                "level": 1,
                "content": "前言\\n文本",
                "char_start": 0,
                "char_end": 5,
                "container_title": None,
                "section_role": "front_matter",
                "parent_chapter_id": None,
                "section_kind": None,
                "is_implicit_section": False,
                "task_units": [],
                "task_artifacts": None,
            }
        ],
    }
    try:
        StructuredDocument.from_dict(legacy_payload)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "normal from_dict should fail-fast for legacy sections-only payloads"
        )
    document = StructuredDocument.from_legacy_dict_for_migration(legacy_payload)
    _assert(
        len(document.sections) == 1,
        "legacy migration loader should keep reading sections-only payloads",
    )
    _assert(
        document.sections[0].section_id == "section-legacy-0",
        "legacy section id should be preserved when loading old payload",
    )


def test_structure_nodes_legacy_load_compatibility() -> None:
    payload = {
        "document_id": "legacy-node-doc",
        "title": "Legacy Node",
        "source_path": None,
        "language": "zh",
        "raw_text": "第一章\n內容",
        "chapters": [
            {
                "chapter_id": "chapter-0",
                "title": "第一章",
                "level": 1,
                "chapter_role": "main_body",
                "sections": [],
            }
        ],
        "structure_nodes": [
            {
                "node_id": "node-0",
                "node_type": "chapter",
                "title": "第一章",
                "level": 1,
                "content": "第一章\n內容",
                "char_start": 0,
                "char_end": 4,
                "section_role": "main_body",
                "children": [],
                "task_units": [],
                "task_artifacts": None,
                "source_section_ids": [],
            }
        ],
        "document_task_artifacts": None,
    }
    document = StructuredDocument.from_json(json.dumps(payload, ensure_ascii=False))
    _assert(document.structure_nodes == [], "normal from_json should ignore legacy structure_nodes")
    legacy_loaded = StructuredDocument.from_legacy_json_for_migration(
        json.dumps(payload, ensure_ascii=False)
    )
    _assert(len(legacy_loaded.structure_nodes) == 1, "migration loader should keep old structure_nodes payload")
    persisted = json.loads(document.to_json())
    _assert(
        "structure_nodes" not in persisted,
        "re-saved JSON should omit structure_nodes by default",
    )


def test_builder_does_not_generate_structure_nodes() -> None:
    flat_document = _build_legacy_sections_only_document()
    built = DocumentHierarchyBuilder().build(flat_document)
    _assert(built.chapters, "builder should produce chapters")
    _assert(built.structure_nodes == [], "builder should not generate structure_nodes mirror")
    _assert(built.sections == [], "builder should clear root sections mirror")


def test_task_unit_object_only_under_chapters() -> None:
    document = _build_hierarchy_document()
    payload = json.loads(document.to_json())
    task_unit_paths = _task_unit_object_paths(payload)
    _assert(task_unit_paths, "fixture should contain task unit objects")
    for path in task_unit_paths:
        _assert(
            path.startswith("chapters[") and ".sections[" in path and ".task_units[" in path,
            f"task unit object must live under chapters[].sections[].task_units[]: {path}",
        )


def test_repository_fills_parent_section_id_on_section_update() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        document = _build_hierarchy_document()
        Path(temp_dir, "cleanup-doc.structured.json").write_text(
            document.to_json(),
            encoding="utf-8",
        )
        repository.update_section_task_units(
            doc_name="cleanup-doc",
            section_id="section-1",
            task_units=[
                TaskUnit(
                    unit_id="external-u0",
                    title="external",
                    container_title=None,
                    content="章節內容",
                    source_section_ids=["section-1"],
                    is_fallback_generated=False,
                    parent_section_id=None,
                )
            ],
        )
        reloaded = repository.load_document("cleanup-doc")
        unit = reloaded.chapters[0].sections[0].task_units[0]
        _assert(
            unit.parent_section_id == "section-1",
            "update_section_task_units should fill parent_section_id",
        )


def test_repository_write_fails_fast_for_legacy_sections_only_document() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        legacy_document = _build_legacy_sections_only_document()
        legacy_document = replace(
            legacy_document,
            structure_nodes=[],
        )
        Path(temp_dir, "legacy-only-doc.structured.json").write_text(
            legacy_document.to_json(
                include_legacy_sections=True,
                include_legacy_structure_nodes=True,
            ),
            encoding="utf-8",
        )
        try:
            repository.update_section_task_units(
                doc_name="legacy-only-doc",
                section_id="section-1",
                task_units=[
                    TaskUnit(
                        unit_id="migrated-u0",
                        title="migrated",
                        container_title=None,
                        content="第一章\n正文",
                        source_section_ids=["section-1"],
                        is_fallback_generated=False,
                        parent_section_id=None,
                    )
                ],
            )
        except ValueError as error:
            _assert(
                "requires explicit migration" in str(error),
                "legacy sections-only repository write should fail-fast with migration guidance",
            )
        else:
            raise AssertionError(
                "legacy sections-only repository write should fail-fast in normal path"
            )


def test_real_document_smoke() -> None:
    if os.getenv("RUN_REAL_DOC_SMOKE") != "1":
        return
    base_url = os.getenv("DEEP_READER_BASE_URL", "http://127.0.0.1:8000")
    health = _get_json(f"{base_url}/health")
    _assert(health.get("status") == "ok", "health check should pass before real smoke")

    prepare_response = _post_json(
        f"{base_url}/documents/prepare",
        {
            "doc_name": "许三观卖血记",
            "mode": "base",
            "force_rebuild": True,
            "structured_parser_mode": "common",
        },
    )
    _assert(bool(prepare_response.get("success")), "real smoke prepare should succeed")

    structured_path = Path("data/structured/许三观卖血记.structured.json")
    prepared_payload = json.loads(structured_path.read_text(encoding="utf-8"))
    _assert(prepared_payload.get("chapters"), "real smoke JSON should contain chapters")
    _assert("sections" not in prepared_payload, "real smoke JSON should not persist root sections")
    _assert(
        "structure_nodes" not in prepared_payload,
        "real smoke JSON should not persist root structure_nodes",
    )
    _assert(
        any(
            chapter.get("chapter_role") == "front_matter"
            for chapter in prepared_payload.get("chapters", [])
        ),
        "real smoke hierarchy should contain front matter chapter",
    )

    layout_response = _post_json(
        f"{base_url}/documents/task-layout",
        {
            "doc_name": "许三观卖血记",
            "refresh_task_units": True,
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
        },
    )
    _assert(layout_response.get("chapters"), "real smoke task-layout should return chapters")

    refreshed_payload = json.loads(structured_path.read_text(encoding="utf-8"))
    _assert("sections" not in refreshed_payload, "refreshed JSON should not persist root sections")
    _assert(
        "structure_nodes" not in refreshed_payload,
        "refreshed JSON should not persist root structure_nodes",
    )
    _assert(
        all(
            task_unit.get("parent_section_id") == section.get("section_id")
            for chapter in refreshed_payload.get("chapters", [])
            for section in chapter.get("sections", [])
            for task_unit in section.get("task_units", [])
        ),
        "refreshed hierarchy task units should contain parent_section_id",
    )
    paths = _task_unit_object_paths(refreshed_payload)
    _assert(paths, "refreshed hierarchy should contain task units")
    _assert(
        all(
            path.startswith("chapters[") and ".sections[" in path and ".task_units[" in path
            for path in paths
        ),
        "real smoke task units should only exist under chapters[].sections[].task_units[]",
    )


def main() -> None:
    test_new_json_has_no_legacy_mirror()
    test_to_dict_omits_root_sections_even_when_chapters_empty_by_default()
    test_to_dict_includes_root_sections_when_explicitly_enabled()
    test_from_dict_fails_for_legacy_sections_only_payload()
    test_structure_nodes_legacy_load_compatibility()
    test_builder_does_not_generate_structure_nodes()
    test_task_unit_object_only_under_chapters()
    test_repository_fills_parent_section_id_on_section_update()
    test_repository_write_fails_fast_for_legacy_sections_only_document()
    test_real_document_smoke()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "no_root_sections_or_structure_nodes",
                    "to_dict_omits_root_sections_even_when_chapters_empty_by_default",
                    "to_dict_includes_root_sections_when_explicitly_enabled",
                    "from_dict_fails_for_legacy_sections_only_payload",
                    "old_structure_nodes_load_compatible",
                    "builder_no_structure_nodes",
                    "task_units_only_under_chapters",
                    "update_section_task_units_fills_parent",
                    "repository_write_fails_fast_for_legacy_sections_only",
                    "real_document_smoke_optional",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
