#!/usr/bin/env python3
"""Compatibility tests for legacy structure_profile payloads."""

from __future__ import annotations

from config.faiss_storage_config import FaissStorageConfig
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _legacy_structure_profile_payload() -> dict[str, object]:
    return {
        "topic": "essay",
        "summary": "A compact essay.",
        "document_language": "zh",
        "structure_profile": {
            "profile_version": "parser_hints_v1",
            "document_language": "zh",
            "structure_type": "essay",
            "structure_level_count": 1,
            "parser_mode_hint": "llm_enhanced",
            "heading_rules": {
                "chapter": {
                    "enabled": True,
                    "keywords": ["第一点"],
                    "regex_candidates": ["第[一二三四五六七八九十]+点"],
                    "positions": [100],
                    "line_anchor_window": 2,
                },
                "section": {
                    "enabled": False,
                    "keywords": [],
                    "regex_candidates": [],
                    "positions": [],
                    "line_anchor_window": 0,
                },
            },
            "regions": {
                "front_matter": {"exists": True, "ranges": [{"start_char": 0, "end_char": 200}]},
                "back_matter": {"exists": False, "ranges": []},
            },
        },
    }


def test_old_profile_without_structure_profile_still_loads() -> None:
    payload = {
        "topic": "Novel",
        "summary": "A novel.",
        "document_language": "en",
    }
    profile = DocumentProfile.from_dict(payload)
    _assert(profile.structure_profile is None, "old profile should load without structure_profile")
    _assert(profile.parser_metadata is None, "old profile should load without parser_metadata")


def test_legacy_structure_profile_round_trip_compatibility() -> None:
    profile = DocumentProfile.from_dict(_legacy_structure_profile_payload())
    _assert(profile.structure_profile is not None, "legacy structure_profile should load")
    payload = profile.to_dict()
    restored = DocumentProfile.from_dict(payload)
    _assert(restored.structure_profile is not None, "legacy structure_profile should persist")
    _assert(
        restored.structure_profile.structure_type == "essay",
        "legacy structure_type should persist",
    )


def test_profile_store_keeps_legacy_structure_profile() -> None:
    profile = DocumentProfile.from_dict(_legacy_structure_profile_payload())
    namespace = "profile-store-legacy-structure-compat"
    config = FaissStorageConfig(namespace=namespace)
    try:
        DocumentProfileStore.save(profile, config)
        restored = DocumentProfileStore.load(config)
        _assert(
            restored.structure_profile is not None,
            "profile store should keep legacy structure_profile",
        )
    finally:
        DocumentProfileStore.clear(config)


if __name__ == "__main__":
    test_old_profile_without_structure_profile_still_loads()
    test_legacy_structure_profile_round_trip_compatibility()
    test_profile_store_keeps_legacy_structure_profile()
    print("test_document_profile_structure_profile: ok")
