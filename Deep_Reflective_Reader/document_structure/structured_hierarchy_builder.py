from __future__ import annotations

import re
from dataclasses import replace

from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_hierarchy import StructuredDocumentNode, StructuredNodeType
from shared.task_unit_model import TaskUnit

_EN_CHAPTER_HEADING_PATTERN = re.compile(
    r"^(?:chapter|chap\.)\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_ZH_CHAPTER_HEADING_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万两兩〇零0-9]+章\b"
)
_EN_SUBSECTION_PATTERN = re.compile(r"^\d+(?:\.\d+)+\b")
_ZH_SUBSECTION_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万两兩〇零0-9]+(?:节|節|部|篇)\b"
)


class DocumentHierarchyBuilder:
    """Build document->chapter->section->task-unit hierarchy from flat sections."""

    _SPECIAL_CHAPTER_LABELS: dict[str, str] = {
        "front_matter": "Front Matter",
        "toc": "Table of Contents",
        "appendix": "Appendix",
        "back_matter": "Back Matter",
        "unknown_region": "Other Sections",
    }

    def build(self, document: StructuredDocument) -> StructuredDocument:
        if not document.sections:
            return replace(document, chapters=[], structure_nodes=[], sections=[])

        mutable_chapters: list[_MutableChapter] = []
        current_chapter: _MutableChapter | None = None
        current_special_chapter: _MutableChapter | None = None
        special_chapter_counts: dict[str, int] = {}

        for section in document.sections:
            role = section.section_role
            title = (section.title or "").strip()
            is_main_body = role in (None, SectionRole.MAIN_BODY)

            if is_main_body and _is_chapter_heading(title):
                current_special_chapter = None
                chapter_id = f"chapter-{len(mutable_chapters)}"
                chapter_section = self._with_section_parent(
                    section=section,
                    parent_chapter_id=chapter_id,
                    section_kind="chapter_body",
                    is_implicit_section=True,
                )
                current_chapter = _MutableChapter(
                    chapter_id=chapter_id,
                    title=chapter_section.title,
                    level=max(1, int(chapter_section.level)),
                    chapter_role=(
                        None
                        if chapter_section.section_role is None
                        else chapter_section.section_role.value
                    ),
                    sections=[chapter_section],
                )
                mutable_chapters.append(current_chapter)
                continue

            if (
                current_chapter is not None
                and is_main_body
                and _should_attach_to_chapter(
                    chapter=current_chapter,
                    section=section,
                )
            ):
                subsection = self._with_section_parent(
                    section=section,
                    parent_chapter_id=current_chapter.chapter_id,
                    section_kind="subsection",
                    is_implicit_section=False,
                )
                current_chapter.sections.append(subsection)
                current_chapter.sections[0] = replace(
                    current_chapter.sections[0],
                    is_implicit_section=False,
                )
                continue

            if is_main_body:
                current_special_chapter = None
                fallback_chapter = self._ensure_main_body_fallback_chapter(
                    mutable_chapters=mutable_chapters,
                    section=section,
                )
                standalone_main_body = self._with_section_parent(
                    section=section,
                    parent_chapter_id=fallback_chapter.chapter_id,
                    section_kind="chapter_body" if not fallback_chapter.sections else "subsection",
                    is_implicit_section=not fallback_chapter.sections,
                )
                fallback_chapter.sections.append(standalone_main_body)
                current_chapter = fallback_chapter
                continue

            current_chapter = None
            special_role = _resolve_special_chapter_role(section)
            if (
                current_special_chapter is None
                or current_special_chapter.chapter_role != special_role
            ):
                special_index = special_chapter_counts.get(special_role, 0)
                special_chapter_counts[special_role] = special_index + 1
                current_special_chapter = _MutableChapter(
                    chapter_id=f"{special_role.replace('_', '-')}-{special_index}",
                    title=self._SPECIAL_CHAPTER_LABELS.get(special_role, "Other Sections"),
                    level=max(1, int(section.level)),
                    chapter_role=special_role,
                    sections=[],
                )
                mutable_chapters.append(current_special_chapter)

            standalone = self._with_section_parent(
                section=section,
                parent_chapter_id=current_special_chapter.chapter_id,
                section_kind=_resolve_special_section_kind(section),
                is_implicit_section=False,
            )
            current_special_chapter.sections.append(standalone)

        chapters = [
            StructuredChapter(
                chapter_id=chapter.chapter_id,
                title=chapter.title,
                level=chapter.level,
                chapter_role=chapter.chapter_role,
                sections=list(chapter.sections),
                task_artifacts=chapter.task_artifacts,
                metadata=dict(chapter.metadata),
            )
            for chapter in mutable_chapters
        ]

        return replace(
            document,
            chapters=chapters,
            structure_nodes=[],
            sections=[],
        )

    @staticmethod
    def _ensure_main_body_fallback_chapter(
        *,
        mutable_chapters: list["_MutableChapter"],
        section: StructuredSection,
    ) -> "_MutableChapter":
        """Create a fallback main-body chapter when flat sections lack explicit chapter headings."""
        chapter_id = f"chapter-{len(mutable_chapters)}"
        chapter_title = section.title or f"Main Body {len(mutable_chapters) + 1}"
        fallback_chapter = _MutableChapter(
            chapter_id=chapter_id,
            title=chapter_title,
            level=max(1, int(section.level)),
            chapter_role=SectionRole.MAIN_BODY.value,
            sections=[],
        )
        mutable_chapters.append(fallback_chapter)
        return fallback_chapter

    @staticmethod
    def _with_section_parent(
        *,
        section: StructuredSection,
        parent_chapter_id: str | None,
        section_kind: str | None,
        is_implicit_section: bool,
    ) -> StructuredSection:
        task_units = [
            replace(task_unit, parent_section_id=section.section_id)
            for task_unit in section.task_units
        ]
        return replace(
            section,
            parent_chapter_id=parent_chapter_id,
            section_kind=section_kind,
            is_implicit_section=is_implicit_section,
            task_units=task_units,
        )

    @staticmethod
    def _build_structure_nodes(
        *,
        sections: list[StructuredSection],
        chapters: list[StructuredChapter],
    ) -> list[StructuredDocumentNode]:
        """Deprecated legacy mirror builder. Keep for backward compatibility only."""
        chapter_by_id = {chapter.chapter_id: chapter for chapter in chapters}
        emitted_chapters: set[str] = set()
        nodes: list[StructuredDocumentNode] = []

        for section in sections:
            chapter_id = section.parent_chapter_id
            if chapter_id is not None:
                if chapter_id in emitted_chapters:
                    continue
                chapter = chapter_by_id.get(chapter_id)
                if chapter is None or not chapter.sections:
                    continue
                chapter_root = chapter.sections[0]
                children = [
                    _build_node_from_section(
                        section=child_section,
                        node_type=StructuredNodeType.SECTION,
                    )
                    for child_section in chapter.sections[1:]
                ]
                nodes.append(
                    StructuredDocumentNode(
                        node_id=f"node::{chapter.chapter_id}",
                        node_type=StructuredNodeType.CHAPTER,
                        title=chapter.title,
                        level=chapter.level,
                        content=chapter_root.content,
                        char_start=chapter_root.char_start,
                        char_end=chapter_root.char_end,
                        section_role=chapter.chapter_role,
                        children=children,
                        task_units=list(chapter_root.task_units),
                        task_artifacts=chapter_root.task_artifacts,
                        source_section_ids=[
                            chapter_section.section_id for chapter_section in chapter.sections
                        ],
                    )
                )
                emitted_chapters.add(chapter_id)
                continue

            nodes.append(
                _build_node_from_section(
                    section=section,
                    node_type=_resolve_node_type(section),
                )
            )
        return nodes


class _MutableChapter:
    def __init__(
        self,
        *,
        chapter_id: str,
        title: str | None,
        level: int,
        chapter_role: str | None,
        sections: list[StructuredSection],
    ) -> None:
        self.chapter_id = chapter_id
        self.title = title
        self.level = level
        self.chapter_role = chapter_role
        self.sections = list(sections)
        self.task_artifacts = None
        self.metadata: dict[str, object] = {
            "legacy_chapter_key": (
                None if title is None else f"chapter::{title.strip()}"
            )
        }


def build_document_hierarchy_from_sections(
    document: StructuredDocument,
) -> StructuredDocument:
    """Backward-compatible function wrapper for hierarchy builder."""
    return DocumentHierarchyBuilder().build(document)


def flattened_sections_from_chapters(
    chapters: list[StructuredChapter],
) -> list[StructuredSection]:
    """Flatten chapter sections without reintroducing legacy root mirrors."""
    flattened: list[StructuredSection] = []
    for chapter in chapters:
        flattened.extend(chapter.sections)
    return flattened


def _build_node_from_section(
    *,
    section: StructuredSection,
    node_type: StructuredNodeType,
) -> StructuredDocumentNode:
    return StructuredDocumentNode(
        node_id=f"node::{section.section_id}",
        node_type=node_type,
        title=section.title,
        level=max(1, int(section.level)),
        content=section.content,
        char_start=section.char_start,
        char_end=section.char_end,
        section_role=(None if section.section_role is None else section.section_role.value),
        children=[],
        task_units=list(section.task_units),
        task_artifacts=section.task_artifacts,
        source_section_ids=[section.section_id],
    )


def _resolve_node_type(section: StructuredSection) -> StructuredNodeType:
    role = section.section_role
    if role == SectionRole.FRONT_MATTER:
        return StructuredNodeType.FRONT_MATTER
    if role == SectionRole.APPENDIX:
        return StructuredNodeType.APPENDIX
    if role == SectionRole.BACK_MATTER:
        return StructuredNodeType.BACK_MATTER
    if role == SectionRole.TOC:
        return StructuredNodeType.FRONT_MATTER
    if _is_chapter_heading((section.title or "").strip()):
        return StructuredNodeType.CHAPTER
    return StructuredNodeType.SECTION


def _resolve_standalone_section_kind(section: StructuredSection) -> str | None:
    role = section.section_role
    if role is None:
        return None
    return role.value


def _resolve_special_chapter_role(section: StructuredSection) -> str:
    role = section.section_role
    if role == SectionRole.FRONT_MATTER:
        return SectionRole.FRONT_MATTER.value
    if role == SectionRole.TOC:
        return SectionRole.TOC.value
    if role == SectionRole.APPENDIX:
        return SectionRole.APPENDIX.value
    if role == SectionRole.BACK_MATTER:
        return SectionRole.BACK_MATTER.value
    return "unknown_region"


def _resolve_special_section_kind(section: StructuredSection) -> str | None:
    role = section.section_role
    if role is None:
        return "unknown_region"
    return role.value


def _is_chapter_heading(title: str) -> bool:
    if not title:
        return False
    return bool(
        _EN_CHAPTER_HEADING_PATTERN.match(title)
        or _ZH_CHAPTER_HEADING_PATTERN.match(title)
    )


def _is_subsection_heading(title: str) -> bool:
    if not title:
        return False
    return bool(
        _EN_SUBSECTION_PATTERN.match(title)
        or _ZH_SUBSECTION_PATTERN.match(title)
    )


def _should_attach_to_chapter(
    *,
    chapter: _MutableChapter,
    section: StructuredSection,
) -> bool:
    title = (section.title or "").strip()
    if _is_subsection_heading(title):
        return True
    return section.level > chapter.level
