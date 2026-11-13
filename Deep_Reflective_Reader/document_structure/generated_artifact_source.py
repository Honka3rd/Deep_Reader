"""Server-owned generation evidence; validation is pure and requires a locked snapshot."""
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

from document_structure.reading_content import ReadingContentConflict, digest, project_document, state_for

BINDING_KEY = "generation_source"
RENAME_EPOCH = "artifact_generation_epoch"


def advance_rename_epoch(document):
    """Called only by an explicit lifecycle owner, before durable publication."""
    return replace(document, parse_provenance={**document.parse_provenance, RENAME_EPOCH: uuid4().hex})


def effective_target(document, level, target_id):
    projected = project_document(document)
    targets = {
        "chapter": [c for c in projected.chapters],
        "section": [s for c in projected.chapters for s in c.sections],
        "task_unit": [u for c in projected.chapters for s in c.sections for u in s.task_units],
    }
    field = {"chapter": "chapter_id", "section": "section_id", "task_unit": "unit_id"}
    matches = [item for item in targets.get(level, []) if getattr(item, field[level]) == target_id]
    if len(matches) != 1:
        raise ReadingContentConflict("generation_target_changed")
    return matches[0]


def effective_section_hash(document, section_id):
    return sha256(effective_target(document, "section", section_id).content.encode("utf-8")).hexdigest()


def capture_generation_source(document, level, target_id):
    provenance = document.parse_provenance
    if not provenance.get("interaction_epoch"):
        raise ReadingContentConflict("generation_incarnation_unknown_explicit_reparse_required")
    effective_target(document, level, target_id)
    projected = project_document(document)
    # Whole consumed hierarchy is conservative: merged units can depend on more
    # than the requested section. Exclude artifacts so independent writes commute.
    source = [(c.chapter_id, c.title, [(s.section_id, s.title, s.content,
        [(u.unit_id, u.title, u.container_title, u.content,
          [(b.block_id, b.content) for b in u.content_blocks]) for u in s.task_units])
        for s in c.sections]) for c in projected.chapters]
    return {"version": 1, "document_id": document.document_id,
            "incarnation": provenance["interaction_epoch"],
            "rename_epoch": provenance.get(RENAME_EPOCH),
            "structure_version": provenance.get("storage_structure_version", 1),
            "reading_revision": state_for(document).get("revision", 0),
            "content_epochs": digest(state_for(document).get("block_content_epochs", {})),
            "level": level, "target_id": target_id, "effective_digest": digest(source)}


def generation_source_matches(document, artifact, level, target_id):
    try:
        return (artifact.metadata or {}).get(BINDING_KEY) == capture_generation_source(document, level, target_id)
    except (ValueError, KeyError, TypeError):
        return False


def require_generation_source(document, artifact, level, target_id):
    if artifact is not None and not generation_source_matches(document, artifact, level, target_id):
        raise ReadingContentConflict("generated_artifact_source_changed")


def require_generated_fields(document, artifacts, level, target_id):
    if artifacts is not None:
        for artifact in (artifacts.summary, artifacts.quiz):
            require_generation_source(document, artifact, level, target_id)
