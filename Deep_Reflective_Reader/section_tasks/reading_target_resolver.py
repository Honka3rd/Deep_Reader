from __future__ import annotations

from dataclasses import dataclass

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_unit_model import TaskUnit


READING_TARGET_LEVELS = frozenset({"document", "book", "chapter", "section", "task_unit"})


@dataclass(frozen=True)
class ResolvedReadingTarget:
    """Hierarchy-resolved reading interaction target."""

    document_id: str
    document_title: str
    target_level: str
    target_id: str
    content: str
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    chapter: StructuredChapter | None = None
    section: StructuredSection | None = None
    task_unit: TaskUnit | None = None


class ReadingTargetResolver:
    """Resolve document/chapter/section/task-unit targets from hierarchy only."""

    def resolve(
        self,
        *,
        document: StructuredDocument,
        target_level: str,
        chapter_id: str | None = None,
        section_id: str | None = None,
        task_unit_id: str | None = None,
    ) -> ResolvedReadingTarget:
        normalized_target_level = self._normalize_target_level(target_level)
        normalized_chapter_id = self._normalize_optional_id(chapter_id)
        normalized_section_id = self._normalize_optional_id(section_id)
        normalized_task_unit_id = self._normalize_optional_id(task_unit_id)

        if not document.chapters:
            raise ValueError(
                "reading target resolution requires chapters hierarchy; "
                "legacy sections/structure_nodes fallback is not supported"
            )

        if normalized_target_level == "document":
            return self._resolve_document_target(
                document=document,
                chapter_id=normalized_chapter_id,
                section_id=normalized_section_id,
                task_unit_id=normalized_task_unit_id,
            )
        if normalized_target_level == "chapter":
            return self._resolve_chapter_target(
                document=document,
                chapter_id=normalized_chapter_id,
                section_id=normalized_section_id,
                task_unit_id=normalized_task_unit_id,
            )
        if normalized_target_level == "section":
            return self._resolve_section_target(
                document=document,
                chapter_id=normalized_chapter_id,
                section_id=normalized_section_id,
                task_unit_id=normalized_task_unit_id,
            )
        return self._resolve_task_unit_target(
            document=document,
            chapter_id=normalized_chapter_id,
            section_id=normalized_section_id,
            task_unit_id=normalized_task_unit_id,
        )

    @staticmethod
    def _normalize_target_level(target_level: str) -> str:
        normalized_target_level = target_level.strip()
        if normalized_target_level == "book":
            normalized_target_level = "document"
        if normalized_target_level not in READING_TARGET_LEVELS:
            raise ValueError(
                "unsupported reading target level: "
                f"'{target_level}'. expected one of: chapter, document, section, task_unit"
            )
        return normalized_target_level

    @staticmethod
    def _normalize_optional_id(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _resolve_document_target(
        self,
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        section_id: str | None,
        task_unit_id: str | None,
    ) -> ResolvedReadingTarget:
        if chapter_id or section_id or task_unit_id:
            raise ValueError("document target must not include child target ids")
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            target_level="document",
            target_id=document.document_id,
            content=document.raw_text,
        )

    def _resolve_chapter_target(
        self,
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        section_id: str | None,
        task_unit_id: str | None,
    ) -> ResolvedReadingTarget:
        if chapter_id is None:
            raise ValueError("chapter target requires chapter_id")
        if section_id or task_unit_id:
            raise ValueError("chapter target must not include section_id or task_unit_id")

        chapter = self._find_unique_chapter(document=document, chapter_id=chapter_id)
        content = "\n\n".join(
            section.content for section in chapter.sections if section.content.strip()
        )
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            target_level="chapter",
            target_id=chapter.chapter_id,
            content=content,
            chapter_id=chapter.chapter_id,
            chapter=chapter,
        )

    def _resolve_section_target(
        self,
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        section_id: str | None,
        task_unit_id: str | None,
    ) -> ResolvedReadingTarget:
        if section_id is None:
            raise ValueError("section target requires section_id")
        if task_unit_id:
            raise ValueError("section target must not include task_unit_id")

        section, parent_chapter = self._find_unique_section(
            document=document,
            section_id=section_id,
        )
        if chapter_id is not None and parent_chapter.chapter_id != chapter_id:
            raise ValueError(
                "section parent chapter mismatch: "
                f"section_id='{section_id}' parent_chapter_id='{parent_chapter.chapter_id}' "
                f"requested_chapter_id='{chapter_id}'"
            )
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            target_level="section",
            target_id=section.section_id,
            content=section.content,
            chapter_id=parent_chapter.chapter_id,
            section_id=section.section_id,
            chapter=parent_chapter,
            section=section,
        )

    def _resolve_task_unit_target(
        self,
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        section_id: str | None,
        task_unit_id: str | None,
    ) -> ResolvedReadingTarget:
        if task_unit_id is None:
            raise ValueError("task_unit target requires task_unit_id")

        task_unit, section, parent_chapter = self._find_unique_task_unit(
            document=document,
            task_unit_id=task_unit_id,
        )
        if section_id is not None and section.section_id != section_id:
            raise ValueError(
                "task_unit parent section mismatch: "
                f"task_unit_id='{task_unit_id}' parent_section_id='{section.section_id}' "
                f"requested_section_id='{section_id}'"
            )
        if chapter_id is not None and parent_chapter.chapter_id != chapter_id:
            raise ValueError(
                "task_unit parent chapter mismatch: "
                f"task_unit_id='{task_unit_id}' parent_chapter_id='{parent_chapter.chapter_id}' "
                f"requested_chapter_id='{chapter_id}'"
            )
        if task_unit.parent_section_id is not None and (
            task_unit.parent_section_id != section.section_id
        ):
            raise ValueError(
                "task_unit hierarchy parent mismatch: "
                f"task_unit_id='{task_unit_id}' parent_section_id='{task_unit.parent_section_id}' "
                f"hierarchy_section_id='{section.section_id}'"
            )
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            target_level="task_unit",
            target_id=task_unit.unit_id,
            content=task_unit.content,
            chapter_id=parent_chapter.chapter_id,
            section_id=section.section_id,
            task_unit_id=task_unit.unit_id,
            chapter=parent_chapter,
            section=section,
            task_unit=task_unit,
        )

    @staticmethod
    def _find_unique_chapter(
        *,
        document: StructuredDocument,
        chapter_id: str,
    ) -> StructuredChapter:
        matches = [
            chapter for chapter in document.chapters
            if chapter.chapter_id == chapter_id
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate_chapter_id:{chapter_id}")
        if not matches:
            raise ValueError(
                f"chapter_id '{chapter_id}' not found in document '{document.document_id}'"
            )
        return matches[0]

    @staticmethod
    def _find_unique_section(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> tuple[StructuredSection, StructuredChapter]:
        matches: list[tuple[StructuredSection, StructuredChapter]] = []
        for chapter in document.chapters:
            for section in chapter.sections:
                if section.section_id == section_id:
                    matches.append((section, chapter))
        if len(matches) > 1:
            raise ValueError(f"duplicate_hierarchy_section_id:{section_id}")
        if not matches:
            raise ValueError(
                f"section_id '{section_id}' not found in document '{document.document_id}'"
            )
        section, chapter = matches[0]
        if section.parent_chapter_id is not None and (
            section.parent_chapter_id != chapter.chapter_id
        ):
            raise ValueError(
                "section hierarchy parent mismatch: "
                f"section_id='{section_id}' parent_chapter_id='{section.parent_chapter_id}' "
                f"hierarchy_chapter_id='{chapter.chapter_id}'"
            )
        return section, chapter

    @staticmethod
    def _find_unique_task_unit(
        *,
        document: StructuredDocument,
        task_unit_id: str,
    ) -> tuple[TaskUnit, StructuredSection, StructuredChapter]:
        matches: list[tuple[TaskUnit, StructuredSection, StructuredChapter]] = []
        for chapter in document.chapters:
            for section in chapter.sections:
                for task_unit in section.task_units:
                    if task_unit.unit_id == task_unit_id:
                        matches.append((task_unit, section, chapter))
        if len(matches) > 1:
            raise ValueError(f"duplicate_task_unit_id:{task_unit_id}")
        if not matches:
            raise ValueError(
                f"task_unit_id '{task_unit_id}' not found in document '{document.document_id}'"
            )
        return matches[0]
