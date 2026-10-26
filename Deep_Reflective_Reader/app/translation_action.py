"""Block/profile orchestration; model work never holds a repository transaction."""

from dataclasses import dataclass
from uuid import uuid4

from config.faiss_storage_config import FaissStorageConfig
from document_structure.document_artifact_repository import DocumentArtifactRepository
from profile.document_profile_store import DocumentProfileStore
from section_tasks.reading_target_resolver import ResolvedReadingTarget
from section_tasks.translation_contracts import TranslationError, TranslationResult, ensure_distinct_translation_languages
from section_tasks.translation_records import (
    commit_translation, interactive_language, replay_record, translation_command,
    translation_languages, translation_snapshot,
)
from section_tasks.translation_service import TranslationService


@dataclass(frozen=True)
class TranslationActionResult:
    target: ResolvedReadingTarget
    translation: TranslationResult
    effective_content_hash: str
    record: dict | None = None


class TranslationAction:
    def __init__(self, repository: DocumentArtifactRepository, profile_store: DocumentProfileStore,
                 translator: TranslationService):
        self._repository = repository
        self._profiles = profile_store
        self._translator = translator

    def generate(self, *, doc_name: str, content_block_id: str, target_language: str,
                 source_language: str | None = None, mutation_id: str | None = None) -> TranslationActionResult:
        target_language = interactive_language(target_language, field="target_language")
        snapshot = translation_snapshot(self._repository.load_document(doc_name), content_block_id)
        if source_language is not None:
            source_language = interactive_language(source_language, field="source_language")
        mutation_id = uuid4().hex if mutation_id is None else mutation_id
        if not isinstance(mutation_id, str) or not 1 <= len(mutation_id.strip()) <= 100:
            raise TranslationError("translation_request_invalid")
        source_language = (self._default_language(doc_name) if source_language is None
                           else source_language)
        ensure_distinct_translation_languages(source_language, target_language)
        saved = next((record for record in snapshot.records()
                      if record["target_language"] == target_language), None)
        if saved is not None:
            result = TranslationResult(saved["source_language"], saved["target_language"],
                                       saved["status"], saved["translated_text"])
            return TranslationActionResult(snapshot.target, result, snapshot.content_hash, saved)
        if source_language is None:
            raise TranslationError("translation_source_language_required")
        command = translation_command(source_language, target_language)
        record = replay_record(snapshot, mutation_id, command)
        if record is not None:
            result = TranslationResult(record["source_language"], record["target_language"],
                                       record["status"], record["translated_text"])
            return TranslationActionResult(snapshot.target, result, snapshot.content_hash, record)
        if len(snapshot.entry.get("receipts", {})) >= 256:
            raise TranslationError("translation_history_limit")
        result = self._translator.translate(text=snapshot.target.content, source_language=source_language,
                                            target_language=target_language)
        if result.status == "untranslatable":
            try:
                current = translation_snapshot(self._repository.load_document(doc_name), content_block_id)
            except (FileNotFoundError, TranslationError):
                raise TranslationError("translation_source_changed") from None
            if current.binding != snapshot.binding or current.target.reading_revision != snapshot.target.reading_revision:
                raise TranslationError("translation_source_changed")
            return TranslationActionResult(snapshot.target, result, snapshot.content_hash)
        try:
            record = self._repository.mutate_document_artifacts(doc_name, lambda current: commit_translation(
                current, expected=snapshot, result=result, mutation_id=mutation_id, command=command))
        except FileNotFoundError:
            raise TranslationError("translation_source_changed") from None
        result = TranslationResult(record["source_language"], record["target_language"],
                                   record["status"], record["translated_text"])
        return TranslationActionResult(snapshot.target, result, snapshot.content_hash, record)

    def _default_language(self, doc_name):
        try:
            profile = self._profiles.load(FaissStorageConfig(doc_name, read_only=True))
        except FileNotFoundError:
            return None
        except Exception:
            raise TranslationError("translation_profile_unavailable") from None
        try:
            return interactive_language(profile.document_language, field="source_language")
        except TranslationError:
            return None

    def read(self, *, doc_name, content_block_id):
        snapshot = translation_snapshot(self._repository.load_document(doc_name), content_block_id)
        return {"target": snapshot.target, "language_codes": list(translation_languages()),
            "default_source_language": self._default_language(doc_name),
            "reading_revision": snapshot.target.reading_revision,
            "effective_content_hash": snapshot.content_hash, "translations": snapshot.records()}
