"""Registry-bound block translation records and atomic commit policy."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib

from document_structure.reading_content import digest, original_blocks, state_for
from language.language_profile_registry import LanguageProfileRegistry
from section_tasks.reading_target_resolver import ReadingTargetResolver, ResolvedReadingTarget
from section_tasks.translation_contracts import TranslationError, TranslationResult
from shared.task_artifacts import DocumentTaskArtifacts


PUBLIC_FIELDS = ("source_language", "target_language", "status", "translated_text",
                 "generated_at", "updated_at", "revision", "mutation_id", "effective_content_hash")


def translation_languages():
    return tuple(code for code in LanguageProfileRegistry.get_supported_language_codes() if code != "unknown")


def interactive_language(value, *, field):
    if not isinstance(value, str) or value not in translation_languages():
        raise TranslationError(f"translation_invalid_{field}")
    return value


def public_record(record):
    return {key: record[key] for key in PUBLIC_FIELDS}


@dataclass(frozen=True)
class TranslationSnapshot:
    target: ResolvedReadingTarget
    binding: str
    content_hash: str
    entry: dict

    def records(self):
        return [public_record(record) for _, record in sorted(self.entry.get("languages", {}).items())
                if record.get("binding") == self.binding]


def translation_snapshot(document, block_id):
    matches = [(c, s, u) for c, s, u, b in original_blocks(document) if b.block_id == block_id]
    if not matches:
        raise TranslationError("translation_target_missing")
    if len(matches) != 1:
        raise TranslationError("translation_target_invalid")
    chapter, section, unit = matches[0]
    try:
        target = ReadingTargetResolver().resolve(document=document, target_level="content_block",
            chapter_id=chapter.chapter_id, section_id=section.section_id,
            task_unit_id=unit.unit_id, content_block_id=block_id)
    except ValueError as error:
        code = "translation_target_missing" if "not found" in str(error) else "translation_source_changed"
        raise TranslationError(code) from None
    target = replace(target, source_structure_version=document.parse_provenance.get("storage_structure_version"))
    content_hash = hashlib.sha256(target.content.encode("utf-8")).hexdigest()
    state = state_for(document)
    binding = digest([target.document_id, target.source_structure_version,
        document.parse_provenance.get("interaction_epoch"), chapter.chapter_id, section.section_id,
        unit.unit_id, block_id, content_hash, state.get("block_content_epochs", {}).get(block_id)])
    artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
    entry = artifacts.metadata.get("translations", {}).get(block_id, {})
    if not isinstance(entry, dict) or not isinstance(entry.get("languages", {}), dict):
        raise TranslationError("translation_service_unavailable")
    return TranslationSnapshot(target, binding, content_hash, deepcopy(entry))


def replay_record(snapshot, mutation_id, command):
    receipt = snapshot.entry.get("receipts", {}).get(mutation_id)
    if receipt is None:
        return None
    record = snapshot.entry.get("languages", {}).get(receipt["target_language"])
    if (receipt["command"] != command or record is None or record["mutation_id"] != mutation_id
            or record.get("binding") != snapshot.binding):
        raise TranslationError("translation_mutation_conflict")
    return public_record(record)


def translation_command(source_language, target_language):
    return digest([source_language, target_language])


def commit_translation(document, *, expected, result: TranslationResult, mutation_id, command):
    try:
        current = translation_snapshot(document, expected.target.content_block_id)
    except TranslationError:
        raise TranslationError("translation_source_changed") from None
    if (current.binding != expected.binding
            or current.target.reading_revision != expected.target.reading_revision):
        raise TranslationError("translation_source_changed")
    artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
    replay = replay_record(current, mutation_id, command)
    if replay is not None:
        return artifacts, replay
    language = result.target_language
    previous = current.entry.get("languages", {}).get(language)
    old = expected.entry.get("languages", {}).get(language)
    if previous != old:
        raise TranslationError("translation_mutation_conflict")
    if len(current.entry.get("receipts", {})) >= 256:
        raise TranslationError("translation_history_limit")
    if result.status not in {"completed", "partial"} or not result.translated_text:
        raise TranslationError("translation_invalid_output")
    now = datetime.now(timezone.utc).isoformat()
    record = {"source_language": result.source_language, "target_language": language,
        "status": result.status, "translated_text": result.translated_text,
        "generated_at": now, "updated_at": now, "revision": (previous or {}).get("revision", 0) + 1,
        "mutation_id": mutation_id, "effective_content_hash": current.content_hash, "binding": current.binding}
    target = current.target
    entry = {**current.entry,
        "target": ["content_block", target.target_id, target.chapter_id, target.section_id,
                   target.task_unit_id, target.content_block_id],
        "languages": {**current.entry.get("languages", {}), language: record},
        "receipts": {**current.entry.get("receipts", {}), mutation_id: {
            "command": command, "target_language": language}}}
    translations = {**artifacts.metadata.get("translations", {}), target.content_block_id: entry}
    return replace(artifacts, metadata={**artifacts.metadata, "translations": translations}), public_record(record)
