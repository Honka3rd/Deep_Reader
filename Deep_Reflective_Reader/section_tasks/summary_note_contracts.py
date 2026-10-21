"""Public summary and user-authored note payloads, independent of transport."""

from dataclasses import dataclass
from typing import Any


SUMMARY_SCHEMA_VERSION = "summary_interaction_v1"
SUMMARY_PROMPT_VERSION = "summary_interaction_prompt_v1"
NOTE_SCHEMA_VERSION = "note_interaction_v1"
NOTE_MAX_CHARS = 10_000


@dataclass(frozen=True)
class SummaryPayload:
    headline: str
    body: str
    key_points: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> "SummaryPayload":
        if not isinstance(value, dict) or set(value) - {"headline", "body", "key_points"}:
            raise ValueError("summary output must contain only headline, body, key_points")
        headline, body = value.get("headline"), value.get("body")
        points = value.get("key_points", [])
        if not isinstance(headline, str) or not headline.strip() or len(headline) > 300:
            raise ValueError("summary headline must contain 1..300 characters")
        if not isinstance(body, str) or not body.strip() or len(body) > 12_000:
            raise ValueError("summary body must contain 1..12000 characters")
        if not isinstance(points, list) or len(points) > 12 or any(
            not isinstance(point, str) or not point.strip() or len(point) > 1000
            for point in points
        ):
            raise ValueError("summary key_points must contain at most 12 non-empty strings")
        return cls(headline.strip(), body.strip(), tuple(point.strip() for point in points))

    def to_dict(self) -> dict[str, object]:
        return {"headline": self.headline, "body": self.body, "key_points": list(self.key_points)}


def validate_note_text(text: str) -> str:
    """Count Unicode code points, preserving user whitespace but rejecting blank notes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("note_text_empty")
    if len(text) > NOTE_MAX_CHARS:
        raise ValueError("note_text_too_long")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise ValueError("note_text_invalid_unicode")
    return text


@dataclass(frozen=True)
class NotePayload:
    """One note per resolved target. Revision zero means create, never overwrite."""

    note_id: str
    text: str
    revision: int
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        validate_note_text(self.text)
        if not self.note_id or type(self.revision) is not int or self.revision < 1:
            raise ValueError("invalid_note_identity")
        if not self.created_at or not self.updated_at:
            raise ValueError("note_timestamps_required")

    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))
