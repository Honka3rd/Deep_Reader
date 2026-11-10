"""Pure preference staging for authoritative document rename/delete transactions."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from reading_preferences.commands import command_receipt
from reading_preferences.models import BookSettings, PreferenceConflict, identifier
from reading_preferences.serialization import empty_book, encode_settings, decode_book


@dataclass(frozen=True)
class PreferenceFileWrite:
    path: Path
    payload: str


def stage_document_lifecycle(scope, source, destination, *, document_id, new_document_id=None, mutation_id):
    identifier(document_id)
    source = deepcopy(source)
    receipt = command_receipt(mutation_id, 'document_lifecycle', document_id, new_document_id,
                              source['revision'], destination['revision'] if destination else None)
    if new_document_id is not None:
        identifier(new_document_id)
        if new_document_id == document_id:
            raise PreferenceConflict('preference_rename_same_identity')
        if destination is not None and not destination['deleted']:
            raise PreferenceConflict('preference_destination_exists')
        destination = deepcopy(destination or empty_book())
        destination.update(settings=source['settings'], deleted=False, receipt=receipt,
                           revision=destination['revision'] + 1)
        decode_book(scope, new_document_id, destination)
    source.update(settings=encode_settings(BookSettings()), deleted=True, receipt=receipt,
                  revision=source['revision'] + 1)
    decode_book(scope, document_id, source)
    return source, destination
