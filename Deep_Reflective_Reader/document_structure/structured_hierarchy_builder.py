from __future__ import annotations

import re
from dataclasses import replace

from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_hierarchy import StructuredDocumentNode, StructuredNodeType

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


class _MutableNode:
    def __init__(self, node: StructuredDocumentNode):
        self.node_id = node.node_id
        self.node_type = node.node_type
        self.title = node.title
        self.level = node.level
        self.content = node.content
        self.char_start = node.char_start
        self.char_end = node.char_end
        self.section_role = node.section_role
        self.children: list[_MutableNode] = []
        self.task_units = list(node.task_units)
        self.task_artifacts = node.task_artifacts
        self.source_section_ids = list(node.source_section_ids)

    def freeze(self) -> StructuredDocumentNode:
        return StructuredDocumentNode(
            node_id=self.node_id,
            node_type=self.node_type,
            title=self.title,
            level=self.level,
            content=self.content,
            char_start=self.char_start,
            char_end=self.char_end,
            section_role=self.section_role,
            children=[child.freeze() for child in self.children],
            task_units=list(self.task_units),
            task_artifacts=self.task_artifacts,
            source_section_ids=list(self.source_section_ids),
        )


def build_document_hierarchy_from_sections(
    document: StructuredDocument,
) -> StructuredDocument:
    """Derive backward-compatible structure_nodes tree from flat sections."""
    if not document.sections:
        return replace(document, structure_nodes=[])

    roots: list[_MutableNode] = []
    current_chapter: _MutableNode | None = None

    for section in document.sections:
        node_type = _resolve_node_type(section)
        node = _MutableNode(_build_node_from_section(section=section, node_type=node_type))

        if node_type == StructuredNodeType.CHAPTER:
            roots.append(node)
            current_chapter = node
            continue

        if (
            node_type == StructuredNodeType.SECTION
            and current_chapter is not None
            and _should_attach_to_chapter(parent=current_chapter, section=section)
        ):
            current_chapter.children.append(node)
            continue

        roots.append(node)
        if node_type != StructuredNodeType.SECTION:
            current_chapter = None

    return replace(document, structure_nodes=[node.freeze() for node in roots])


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
        section_role=None if section.section_role is None else section.section_role.value,
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

    title = (section.title or "").strip()
    if _is_chapter_heading(title):
        return StructuredNodeType.CHAPTER
    return StructuredNodeType.SECTION


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
    parent: _MutableNode,
    section: StructuredSection,
) -> bool:
    """Attach section beneath chapter only when it looks like true subdivision."""
    title = (section.title or "").strip()
    if not _is_subsection_heading(title):
        return False
    return section.level > parent.level
