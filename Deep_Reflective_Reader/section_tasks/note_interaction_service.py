"""Explicit, atomic single-note mutations. No model or preparation dependencies."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.summary_note_contracts import NotePayload, validate_note_text
from shared.task_artifacts import DocumentTaskArtifacts


class NoteConflict(ValueError):
    pass


class NoteInteractionService:
    def __init__(self, repository):
        self.repository = repository
        self.resolver = ReadingTargetResolver()

    @staticmethod
    def _identity(target):
        return [target.target_level, target.target_id, target.chapter_id,
                target.section_id, target.task_unit_id, target.content_block_id]

    def _state(self, document, target_options):
        supplied = target_options.get("source_structure_version")
        if supplied is not None and supplied != document.parse_provenance.get("storage_structure_version"):
            raise NoteConflict("note_target_stale")
        target = self.resolver.resolve(document=document, apply_edits=False, **target_options)
        key = hashlib.sha256(json.dumps(self._identity(target)).encode()).hexdigest()
        artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
        notes = dict(artifacts.metadata.get("note", {}))
        entry = notes.get(key, {})
        # Exclude titles: label-only repairs do not invalidate a note editor.
        structure = [(chapter.chapter_id, [(section.section_id,
                      [(unit.unit_id, unit.content) for unit in section.task_units])
                      for section in chapter.sections]) for chapter in document.chapters]
        source = hashlib.sha256(json.dumps([
            document.document_id, document.parse_provenance.get("storage_structure_version"),
            document.parse_provenance.get("interaction_epoch"), structure,
            self._identity(target), target.content,
        ], ensure_ascii=False).encode()).hexdigest()
        token = hashlib.sha256((source + entry.get("epoch", "")).encode()).hexdigest()
        if entry.get("note") and entry.get("source") != source:
            raise NoteConflict("note_target_stale")
        return target, key, artifacts, notes, entry, source, token

    def read(self, doc_name, **target_options):
        state = self._state(self.repository.load_document(doc_name), target_options)
        return self._response(state)

    @staticmethod
    def _response(state):
        target, _, _, _, entry, _, token = state
        return {"target": target, "target_token": token, "note": entry.get("note"),
                "max_chars": 10000,"source_stale":target.reading_revision > 0}

    def save(self, doc_name, *, text, expected_revision, mutation_id, target_token,
             **target_options):
        validate_note_text(text)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("invalid_note_revision")
        if not mutation_id or len(mutation_id) > 100:
            raise ValueError("invalid_note_mutation_id")

        def mutation(document):
            try:
                state = self._state(document, target_options)
            except ValueError as error:
                raise NoteConflict("note_target_stale") from error
            target, key, artifacts, notes, entry, source, token = state
            if token != target_token:
                raise NoteConflict("note_target_stale")
            existing = entry.get("note")
            if entry.get("mutation_id") == mutation_id:
                if existing and existing["text"] == text and entry.get("expected_revision") == expected_revision:
                    return artifacts, self._response(state)
                raise NoteConflict("note_mutation_conflict")
            if (existing["revision"] if existing else 0) != expected_revision:
                raise NoteConflict("note_revision_conflict")
            now = datetime.now(timezone.utc).isoformat()
            note = NotePayload(existing["note_id"] if existing else uuid4().hex, text,
                               expected_revision + 1,
                               existing["created_at"] if existing else now, now)
            notes[key] = {"note": note.to_dict(), "source": source,
                          "epoch": entry.get("epoch", ""), "mutation_id": mutation_id,
                          "expected_revision": expected_revision,
                          "target": self._identity(target)}
            updated = replace(artifacts, metadata={**artifacts.metadata, "note": notes})
            return updated, {**self._response(state), "note": note.to_dict()}

        return self.repository.mutate_document_artifacts(doc_name, mutation)

    def delete(self, doc_name, *, expected_revision, target_token, **target_options):
        def mutation(document):
            try:
                state = self._state(document, target_options)
            except ValueError as error:
                raise NoteConflict("note_target_stale") from error
            _, key, artifacts, notes, entry, _, token = state
            if token != target_token:
                raise NoteConflict("note_target_stale")
            existing = entry.get("note")
            if not existing or existing["revision"] != expected_revision:
                raise NoteConflict("note_revision_conflict")
            # A new epoch invalidates delayed creates from before the deletion.
            notes[key] = {"epoch": uuid4().hex}
            updated = replace(artifacts, metadata={**artifacts.metadata, "note": notes})
            return updated, self._response(self._state(
                replace(document, document_task_artifacts=updated), target_options))

        return self.repository.mutate_document_artifacts(doc_name, mutation)
