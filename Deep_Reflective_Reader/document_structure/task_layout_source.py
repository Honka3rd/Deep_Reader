"""Source evidence for explicit layout publication, validated on a locked snapshot."""
from dataclasses import replace
from document_structure.reading_content import ReadingContentConflict, digest, project_document

LAYOUT_SOURCE_KEY = "publication_source"


def _without_artifacts(value):
    if isinstance(value, dict):
        return {key: _without_artifacts(item) for key, item in value.items()
                if key not in {"task_artifacts", "document_task_artifacts", "artifact_ids",
                               "artifact_target_refs", "interaction_availability"}}
    if isinstance(value, list):
        return [_without_artifacts(item) for item in value]
    return value


def same_unit_source(left, right):
    """Compare within one owning section; allow ID/artifact/derived-parent differences."""
    def source(unit):
        return _without_artifacts(replace(unit.with_unit_id("layout-unit"),
            parent_section_id=None).to_dict(include_content_blocks=True))
    return source(left) == source(right)


def capture_layout_source(document) -> str:
    """Bind all consumed source/identity, excluding independently saved artifacts."""
    if not document.parse_provenance.get("interaction_epoch"):
        raise ReadingContentConflict("layout_incarnation_unknown_explicit_reparse_required")

    def source_snapshot(view):
        payload = view.to_dict()
        provenance = dict(payload["parse_provenance"])
        state = provenance.pop("reading_content", {})
        # No-op receipts and annotation audit data do not change resolver inputs.
        payload["parse_provenance"] = provenance
        payload["reading_source"] = {key: state.get(key, default) for key, default in (
            ("revision", 0), ("block_content_epochs", {}), ("blocks", {}),
            ("cleanup_spans", []), ("source_edits", []), ("cleanup_source_sha256", None),
            ("standalone_cleanup_receipts", []))}
        for chapter, chapter_payload in zip(view.chapters, payload["chapters"]):
            chapter_payload.pop("metadata", None)  # advisory/extensions are not resolver source
            for section, section_payload in zip(chapter.sections, chapter_payload["sections"]):
                section_payload["task_units"] = [unit.to_dict(include_content_blocks=True)
                                                for unit in section.task_units]
        return _without_artifacts(payload)

    return digest({"version": 1, "canonical": source_snapshot(document),
                   "effective": source_snapshot(project_document(document))})


def require_layout_source(document, metadata):
    if metadata.get(LAYOUT_SOURCE_KEY) != capture_layout_source(document):
        raise ReadingContentConflict("task_layout_source_changed")
