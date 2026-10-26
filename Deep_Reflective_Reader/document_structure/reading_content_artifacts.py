"""Artifact lifecycle for explicit block edits, separate from overlay geometry."""

from dataclasses import replace

from shared.task_artifacts import DocumentTaskArtifacts


def clear_source_dependents(document, block_ids):
    """Unknown legacy summary dependencies require summary-only document invalidation."""
    if not block_ids:
        return document
    artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
    metadata = {key: value for key, value in artifacts.metadata.items()
                if key not in {"public_summary", "ancestor_summary_artifacts"}}
    if "translations" in metadata:
        metadata["translations"] = {key: value for key, value in metadata["translations"].items()
                                    if key not in block_ids}
    def without_summary(value):
        return replace(value, summary=None) if value is not None else None
    return replace(document, document_task_artifacts=replace(artifacts, metadata=metadata,
        chapter_artifacts={key: without_summary(value) for key, value in artifacts.chapter_artifacts.items()}),
        chapters=[replace(c, task_artifacts=without_summary(c.task_artifacts), sections=[
            replace(s, task_artifacts=without_summary(s.task_artifacts), task_units=[
                replace(u, task_artifacts=without_summary(u.task_artifacts)) for u in s.task_units])
            for s in c.sections]) for c in document.chapters])


def apply_artifact_policy(document, *, invalidate_generated, cleared_block_id=None, changed_block_ids=()):
    document = clear_source_dependents(document, changed_block_ids)
    if invalidate_generated:
        from document_structure.ocr_cleanup_overlay import invalidate_generated as invalidate
        document = invalidate(document)
    if cleared_block_id is None:
        return document
    artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
    notes = artifacts.metadata.get("note", {})
    # Note target identity is [level, id, chapter, section, unit, block].
    retained = {key: entry for key, entry in notes.items() if not (
        isinstance(entry, dict) and isinstance(entry.get("target"), list)
        and len(entry["target"]) == 6 and entry["target"][0] == "content_block"
        and entry["target"][5] == cleared_block_id)}
    return replace(document, document_task_artifacts=replace(artifacts,
        metadata={**artifacts.metadata, "note": retained}))
