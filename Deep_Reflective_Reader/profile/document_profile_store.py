import json
import os

from profile.document_profile import DocumentProfile
from storage_config import StorageConfig


class DocumentProfileStore:
    @staticmethod
    def save(profile: DocumentProfile, config: StorageConfig) -> None:
        payload = {
            "topic": profile.topic,
            "summary": profile.summary,
            "document_language": profile.document_language,
        }

        with open(config.get_raw_profile_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(config: StorageConfig) -> DocumentProfile:
        with open(config.get_raw_profile_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        return DocumentProfile(
            topic=payload["topic"],
            summary=payload["summary"],
            document_language=payload["document_language"],
        )

    @staticmethod
    def exists(config: StorageConfig) -> bool:
        return os.path.exists(config.get_raw_profile_path())

    @staticmethod
    def clear(config: StorageConfig) -> None:
        path = config.get_raw_profile_path()
        if os.path.exists(path):
            os.remove(path)