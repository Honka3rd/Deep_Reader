import json
from dataclasses import dataclass, field
from typing import Any

from document_structure.section_role import SectionRole
from document_structure.structured_hierarchy import StructuredDocumentNode
from shared.task_artifacts import DocumentTaskArtifacts, TaskArtifacts
from shared.task_unit_model import TaskUnit


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
    parent_chapter_id: str | None = None
    section_kind: str | None = None
    is_implicit_section: bool = False
    task_units: list[TaskUnit] = field(default_factory=list)
    task_artifacts: TaskArtifacts | None = None

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
            "parent_chapter_id": self.parent_chapter_id,
            "section_kind": self.section_kind,
            "is_implicit_section": self.is_implicit_section,
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
            "task_artifacts": (
                None if self.task_artifacts is None else self.task_artifacts.to_dict()
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
            parent_chapter_id=(
                None
                if data.get("parent_chapter_id") is None
                else str(data.get("parent_chapter_id"))
            ),
            section_kind=(
                None if data.get("section_kind") is None else str(data.get("section_kind"))
            ),
            is_implicit_section=bool(data.get("is_implicit_section", False)),
            task_units=[
                TaskUnit.from_dict(task_unit_data)
                for task_unit_data in data.get("task_units", [])
                if isinstance(task_unit_data, dict)
            ],
            task_artifacts=TaskArtifacts.from_dict(data.get("task_artifacts")),
        )


@dataclass(frozen=True)
class StructuredChapter:
    """Hierarchy chapter DTO with nested sections for direct tree rendering."""

    chapter_id: str
    title: str | None
    level: int
    chapter_role: str | None
    sections: list[StructuredSection] = field(default_factory=list)
    task_artifacts: TaskArtifacts | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chapter into JSON-friendly dictionary."""
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "level": self.level,
            "chapter_role": self.chapter_role,
            "sections": [section.to_dict() for section in self.sections],
            "task_artifacts": (
                None if self.task_artifacts is None else self.task_artifacts.to_dict()
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredChapter":
        """Deserialize chapter payload with nested sections."""
        return cls(
            chapter_id=str(data["chapter_id"]),
            title=(None if data.get("title") is None else str(data.get("title"))),
            level=int(data.get("level", 1)),
            chapter_role=(
                None
                if data.get("chapter_role") is None
                else str(data.get("chapter_role"))
            ),
            sections=[
                StructuredSection.from_dict(section_payload)
                for section_payload in data.get("sections", [])
                if isinstance(section_payload, dict)
            ],
            task_artifacts=TaskArtifacts.from_dict(data.get("task_artifacts")),
            metadata=(
                {}
                if not isinstance(data.get("metadata"), dict)
                else dict(data.get("metadata"))
            ),
        )


@dataclass(frozen=True)
class StructuredDocument:
    """Top-level structured document DTO.

    `chapters` is the primary hierarchy representation and the only persisted
    section/task-unit source for new JSON artifacts.
    `sections` is deprecated compatibility input for loading legacy payloads.
    `structure_nodes` is legacy experimental hierarchy and should not be expanded.
    """

    document_id: str
    title: str
    source_path: str | None
    language: str | None
    raw_text: str
    sections: list[StructuredSection] = field(default_factory=list)
    chapters: list[StructuredChapter] = field(default_factory=list)
    structure_nodes: list[StructuredDocumentNode] = field(default_factory=list)
    structure_error_code: str | None = None
    structure_error_message: str | None = None
    document_task_artifacts: DocumentTaskArtifacts | None = None

    def to_dict(
        self,
        *,
        include_legacy_sections: bool = False,
        include_legacy_structure_nodes: bool = False,
    ) -> dict[str, Any]:
        """Serialize this document into a JSON-friendly dictionary."""
        payload = {
            "document_id": self.document_id,
            "title": self.title,
            "source_path": self.source_path,
            "language": self.language,
            "raw_text": self.raw_text,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "structure_error_code": self.structure_error_code,
            "structure_error_message": self.structure_error_message,
            "document_task_artifacts": (
                None
                if self.document_task_artifacts is None
                else self.document_task_artifacts.to_dict()
            ),
        }
        if include_legacy_sections:
            payload["sections"] = [section.to_dict() for section in self.sections]
        if include_legacy_structure_nodes:
            payload["structure_nodes"] = [
                node.to_dict() for node in self.structure_nodes
            ]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredDocument":
        """Build a structured document DTO from dictionary payload."""
        section_payloads = data.get("sections", [])
        chapter_payloads = data.get("chapters", [])
        structure_node_payloads = data.get("structure_nodes", [])
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
            chapters=[
                StructuredChapter.from_dict(chapter_payload)
                for chapter_payload in chapter_payloads
                if isinstance(chapter_payload, dict)
            ],
            structure_nodes=[
                StructuredDocumentNode.from_dict(node_data)
                for node_data in structure_node_payloads
                if isinstance(node_data, dict)
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
            document_task_artifacts=DocumentTaskArtifacts.from_dict(
                data.get("document_task_artifacts")
            ),
        )

    def to_json(
        self,
        *,
        ensure_ascii: bool = False,
        indent: int | None = 2,
        include_legacy_sections: bool = False,
        include_legacy_structure_nodes: bool = False,
    ) -> str:
        """Serialize this document into a JSON string."""
        return json.dumps(
            self.to_dict(
                include_legacy_sections=include_legacy_sections,
                include_legacy_structure_nodes=include_legacy_structure_nodes,
            ),
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
