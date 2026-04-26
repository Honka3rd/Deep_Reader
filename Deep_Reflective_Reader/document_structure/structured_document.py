import json
from dataclasses import dataclass, field
from typing import Any

from document_structure.section_role import SectionRole


@dataclass(frozen=True)
class StructuredSection:
    """Flat section DTO for structured-document persistence."""

    section_id: str
    section_index: int
    title: str | None
    level: int
    content: str
    char_start: int
    char_end: int
    container_title: str | None = None
    section_role: SectionRole | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this section into a JSON-friendly dictionary."""
        return {
            "section_id": self.section_id,
            "section_index": self.section_index,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "container_title": self.container_title,
            "section_role": (
                None if self.section_role is None else self.section_role.value
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredSection":
        """Build a section DTO from dictionary payload."""
        return cls(
            section_id=str(data["section_id"]),
            section_index=int(data["section_index"]),
            title=(None if data.get("title") is None else str(data.get("title"))),
            level=int(data["level"]),
            content=str(data["content"]),
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
            container_title=(
                None
                if data.get("container_title") is None
                else str(data.get("container_title"))
            ),
            section_role=SectionRole.resolve(data.get("section_role")),
        )


@dataclass(frozen=True)
class StructuredDocument:
    """Top-level structured document DTO with flat section list."""

    document_id: str
    title: str
    source_path: str | None
    language: str | None
    raw_text: str
    sections: list[StructuredSection] = field(default_factory=list)
    structure_error_code: str | None = None
    structure_error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this document into a JSON-friendly dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_path": self.source_path,
            "language": self.language,
            "raw_text": self.raw_text,
            "sections": [section.to_dict() for section in self.sections],
            "structure_error_code": self.structure_error_code,
            "structure_error_message": self.structure_error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredDocument":
        """Build a structured document DTO from dictionary payload."""
        section_payloads = data.get("sections", [])
        return cls(
            document_id=str(data["document_id"]),
            title=str(data["title"]),
            source_path=(
                None
                if data.get("source_path") is None
                else str(data.get("source_path"))
            ),
            language=(None if data.get("language") is None else str(data.get("language"))),
            raw_text=str(data["raw_text"]),
            sections=[
                StructuredSection.from_dict(section_data)
                for section_data in section_payloads
            ],
            structure_error_code=(
                None
                if data.get("structure_error_code") is None
                else str(data.get("structure_error_code"))
            ),
            structure_error_message=(
                None
                if data.get("structure_error_message") is None
                else str(data.get("structure_error_message"))
            ),
        )

    def to_json(self, *, ensure_ascii: bool = False, indent: int | None = 2) -> str:
        """Serialize this document into a JSON string."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=ensure_ascii,
            indent=indent,
        )

    @classmethod
    def from_json(cls, payload: str) -> "StructuredDocument":
        """Deserialize one structured document from JSON string payload."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("StructuredDocument JSON payload must be an object")
        return cls.from_dict(data)
