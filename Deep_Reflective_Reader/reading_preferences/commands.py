"""Shared receipt identity and revision guards for preference adapters."""

import hashlib

from reading_preferences.models import PreferenceConflict, identifier
from reading_preferences.serialization import canonical_json


def command_receipt(mutation_id, *parts):
    identifier(mutation_id)
    return {"mutation_id": mutation_id,
            "signature": hashlib.sha256(canonical_json(parts).encode()).hexdigest()}


def is_replay(entry, receipt):
    old = entry["receipt"]
    if old is not None and old["mutation_id"] == receipt["mutation_id"]:
        if old != receipt:
            raise PreferenceConflict("preference_mutation_conflict")
        return True
    return False


def check_revision(entry, expected):
    if entry["revision"] != expected:
        raise PreferenceConflict("preference_revision_changed")
