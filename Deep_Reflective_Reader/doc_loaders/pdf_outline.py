"""Native PDF Outline evidence for deterministic structure discovery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PdfOutlineEntry:
    """One bookmark with its resolved PDF destination."""

    title: str
    level: int
    page_index: int | None
    page_label: str | None = None
    destination_type: str | None = None


@dataclass(frozen=True)
class PdfOutlineResult:
    """Validated-or-rejected native bookmark evidence."""

    present: bool
    usable: bool
    entries: list[PdfOutlineEntry] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    source_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "usable": self.usable,
            "entries": [
                {
                    "title": entry.title,
                    "level": entry.level,
                    "page_index": entry.page_index,
                    "page_label": entry.page_label,
                    "destination_type": entry.destination_type,
                }
                for entry in self.entries
            ],
            "reasons": list(self.reasons),
            "source_sha256": self.source_sha256,
        }
