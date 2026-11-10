"""Strict versioned preference codec shared by storage adapters; no read-time migration."""

from dataclasses import asdict
import json

from reading_preferences.models import (
    BookPreferences, BookSettings, LanguageLearningSettings, PreservedModeSettings,
    PreferenceError, WorkspacePreferences, WorkspaceScope, identifier, revision,
)


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PreferenceError("preference_duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value):
    raise PreferenceError("preference_json_constant_invalid")


def exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise PreferenceError("preference_payload_invalid")


def nullable_code(value):
    if value is not None:
        identifier(value)  # Retain old codes on read; the service evaluates eligibility.


def encode_settings(value: BookSettings) -> dict:
    if type(value) is not BookSettings or type(value.modes) is not tuple:
        raise PreferenceError("preference_settings_invalid")
    modes = {}
    for entry in value.modes:
        if type(entry) is LanguageLearningSettings:
            payload = asdict(entry)
            version = payload.pop("version")
        elif type(entry) is PreservedModeSettings:
            version = entry.version
            try:
                payload = json.loads(entry.payload_json, object_pairs_hook=strict_object,
                                     parse_constant=reject_constant)
            except (ValueError, TypeError):
                raise PreferenceError("preference_payload_invalid") from None
        else:
            raise PreferenceError("preference_settings_invalid")
        if entry.mode_id in modes:
            raise PreferenceError("preference_settings_duplicate")
        modes[entry.mode_id] = {"version": version, "payload": payload}
    return {"active_mode": value.active_mode, "source_language_override": value.source_language_override,
            "modes": modes}


def decode_settings(data) -> BookSettings:
    exact_keys(data, ("active_mode", "source_language_override", "modes"))
    identifier(data["active_mode"])
    nullable_code(data["source_language_override"])
    if type(data["modes"]) is not dict:
        raise PreferenceError("preference_payload_invalid")
    modes = []
    for mode_id, entry in sorted(data["modes"].items()):
        identifier(mode_id)
        exact_keys(entry, ("version", "payload"))
        identifier(entry["version"])
        if type(entry["payload"]) is not dict:
            raise PreferenceError("preference_payload_invalid")
        payload = entry["payload"]
        if mode_id == "language_learning" and entry["version"] == "1":
            exact_keys(payload, ("language_code", "scheme_id", "scheme_version", "level_id"))
            for value in payload.values():
                identifier(value)
            modes.append(LanguageLearningSettings(**payload))
        else:
            modes.append(PreservedModeSettings(mode_id, entry["version"], canonical_json(payload)))
    return BookSettings(data["active_mode"], data["source_language_override"], tuple(modes))


def empty_state(scope: WorkspaceScope) -> dict:
    return {"schema_version": 1, "scope": {"kind": scope.kind, "namespace": scope.namespace},
            "workspace": {"revision": 0, "native_language": None, "receipt": None}, "books": {}}


def empty_book() -> dict:
    return {"revision": 0, "settings": encode_settings(BookSettings()), "deleted": False, "receipt": None}


def decode_workspace(scope: WorkspaceScope, data) -> WorkspacePreferences:
    exact_keys(data, ("revision", "native_language", "receipt"))
    revision(data["revision"])
    nullable_code(data["native_language"])
    validate_receipt(data["receipt"])
    return WorkspacePreferences(scope, data["revision"], data["native_language"])


def validate_receipt(data):
    if data is not None:
        exact_keys(data, ("mutation_id", "signature"))
        identifier(data["mutation_id"])
        if (not isinstance(data["signature"], str) or len(data["signature"]) != 64
                or any(c not in "0123456789abcdef" for c in data["signature"])):
            raise PreferenceError("preference_payload_invalid")


def decode_book(scope: WorkspaceScope, document_id: str, data) -> BookPreferences:
    exact_keys(data, ("revision", "settings", "deleted", "receipt"))
    revision(data["revision"])
    if type(data["deleted"]) is not bool:
        raise PreferenceError("preference_payload_invalid")
    validate_receipt(data["receipt"])
    settings = decode_settings(data["settings"])
    if data["deleted"] and settings != BookSettings():
        raise PreferenceError("preference_tombstone_invalid")
    return BookPreferences(scope, document_id, data["revision"], settings, data["deleted"])


def validate_state(scope: WorkspaceScope, data):
    exact_keys(data, ("schema_version", "scope", "workspace", "books"))
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise PreferenceError("preference_storage_version_unsupported")
    if data["scope"] != {"kind": scope.kind, "namespace": scope.namespace}:
        raise PreferenceError("preference_scope_mismatch")
    decode_workspace(scope, data["workspace"])
    if type(data["books"]) is not dict:
        raise PreferenceError("preference_payload_invalid")
    for document_id, entry in data["books"].items():
        identifier(document_id)
        decode_book(scope, document_id, entry)
