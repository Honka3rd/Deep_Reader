"""Rekey interaction identities during an explicit document rename, never prose."""
from dataclasses import replace

from section_tasks.note_interaction_service import NoteInteractionService
from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact


def renamed_interactions(document, new_name):
    old_name = document.document_id
    renamed = replace(document, document_id=new_name,
                      title=new_name if document.title == old_name else document.title)
    artifacts = document.document_task_artifacts
    if artifacts is None:
        return renamed
    metadata = dict(artifacts.metadata)
    notes = {}
    service = NoteInteractionService(None)
    for key, entry in metadata.get("note", {}).items():
        identity = entry.get("target")
        if not identity:
            notes[key] = entry
            continue
        level, _, chapter, section, unit, block = identity
        options = dict(target_level=level, chapter_id=chapter, section_id=section,
                       task_unit_id=unit, content_block_id=block)
        before = service._state(replace(document, document_task_artifacts=None), options)
        after = service._state(replace(renamed, document_task_artifacts=None), options)
        new_key = after[1]
        notes[new_key] = {**entry, "source": after[5] if entry.get("source") == before[5] else entry.get("source"),
                          "target": service._identity(after[0])}
    if "note" in metadata:
        metadata["note"] = notes
    summaries = {}
    for data in metadata.get("public_summary", {}).values():
        artifact = ReadingInteractionArtifact(**data)
        artifact = replace(artifact, document_id=new_name,
                           target_id=new_name if artifact.target_level == "document" else artifact.target_id)
        key = "::".join(["summary", new_name, artifact.target_level, artifact.target_id])
        artifact = replace(artifact, metadata={**artifact.metadata, "artifact_id": key,
                           "context": {**artifact.metadata.get("context", {}), "doc_name": new_name}})
        summaries[key] = artifact.to_dict()
    if "public_summary" in metadata:
        metadata["public_summary"] = summaries
    return replace(renamed, document_task_artifacts=replace(artifacts, metadata=metadata))
