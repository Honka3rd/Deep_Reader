from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from document_structure.text_normalization import normalize_ocr_text
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_unit_model import (
    TaskUnit,
    TaskUnitContentBlock,
    build_default_content_block_id,
)


READING_TARGET_LEVELS = frozenset(
    {"document", "book", "chapter", "section", "task_unit", "content_block"}
)
CONTENT_BLOCK_VALIDATION_READ_MODEL = "segmented_task_unit_content_blocks"


@dataclass(frozen=True)
class ResolvedReadingTarget:
    """Hierarchy-resolved reading interaction target."""

    document_id: str
    document_title: str
    target_level: str
    target_id: str
    content: str
    document_language: str | None = None
    source_structure_version: int | None = None
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    content_block_id: str | None = None
    chapter: StructuredChapter | None = None
    section: StructuredSection | None = None
    task_unit: TaskUnit | None = None
    content_block: TaskUnitContentBlock | None = None
    reading_revision: int = 0
    source_needs_review: bool = False


class ReadingTargetResolver:
    """Resolve document/chapter/section/task-unit targets from hierarchy only."""

    def content_blocks_for_target_validation(
        self,
        task_unit: TaskUnit,
        section: StructuredSection | None = None,
        chapter: StructuredChapter | None = None,
    ) -> list[TaskUnitContentBlock]:
        """
        Return the deterministic read model for content-block target validation.

        The source is the hierarchy-resolved task unit plus the same deterministic
        segmented render projection used by the task-unit content endpoint. Explicit
        custom `TaskUnit.content_blocks` remain accepted when their ids do not collide
        with the segmented render ids. This keeps visible UI block ids addressable
        without using UI row position/title text as authority.
        """
        if any((b.metadata or {}).get("reading_overlay") for b in task_unit.content_blocks):
            return list(task_unit.content_blocks)
        render_blocks = self._build_segmented_content_blocks_for_target_validation(
            task_unit=task_unit,
            section=section,
            chapter=chapter,
        )
        explicit_blocks = task_unit.to_content_blocks()
        if not render_blocks:
            return explicit_blocks

        render_block_ids = {block.block_id for block in render_blocks}
        non_colliding_explicit_blocks = [
            block
            for block in explicit_blocks
            if block.block_id not in render_block_ids
        ]
        return [*render_blocks, *non_colliding_explicit_blocks]

    def resolve(self, *, document, apply_edits=True, **kwargs):
        from document_structure.reading_content import project_document, needs_review, state_for
        resolved = self._resolve(document=project_document(document) if apply_edits else document, **kwargs)
        return replace(resolved, reading_revision=state_for(document).get("revision",0),
                       source_needs_review=needs_review(document))

    def _resolve(
        self,
        *,
        document: StructuredDocument,
        target_level: str,
        chapter_id: str | None = None,
        section_id: str | None = None,
        task_unit_id: str | None = None,
        content_block_id: str | None = None,
        source_hash: str | None = None,
        source_structure_version: int | None = None,
    ) -> ResolvedReadingTarget:
        normalized_target_level = self._normalize_target_level(target_level)
        normalized_chapter_id = self._normalize_optional_id(chapter_id)
        normalized_section_id = self._normalize_optional_id(section_id)
        normalized_task_unit_id = self._normalize_optional_id(task_unit_id)
        normalized_content_block_id = self._normalize_optional_id(content_block_id)
        normalized_source_hash = self._normalize_optional_id(source_hash)

        if normalized_content_block_id is not None:
            if normalized_target_level != "content_block":
                raise ValueError("non-content_block targets must not include content_block_id")
        elif normalized_target_level == "content_block":
            raise ValueError("content_block target requires content_block_id")

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
        if normalized_target_level == "task_unit":
            return self._resolve_task_unit_target(
                document=document,
                chapter_id=normalized_chapter_id,
                section_id=normalized_section_id,
                task_unit_id=normalized_task_unit_id,
            )
        return self._resolve_content_block_target(
            document=document,
            chapter_id=normalized_chapter_id,
            section_id=normalized_section_id,
            task_unit_id=normalized_task_unit_id,
            content_block_id=normalized_content_block_id,
            source_hash=normalized_source_hash,
            source_structure_version=source_structure_version,
        )

    @staticmethod
    def _normalize_target_level(target_level: str) -> str:
        normalized_target_level = target_level.strip()
        if normalized_target_level == "book":
            normalized_target_level = "document"
        if normalized_target_level not in READING_TARGET_LEVELS:
            raise ValueError(
                "unsupported reading target level: "
                f"'{target_level}'. expected one of: chapter, content_block, "
                "document, section, task_unit"
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
            document_language=document.language,
            target_level="document",
            target_id=document.document_id,
            content=self._content_from_document(document),
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
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            document_language=document.language,
            target_level="chapter",
            target_id=chapter.chapter_id,
            content=self._content_from_chapter(chapter),
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
            document_language=document.language,
            target_level="section",
            target_id=section.section_id,
            content=self._content_from_section(section),
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
            document_language=document.language,
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

    def _resolve_content_block_target(
        self,
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        section_id: str | None,
        task_unit_id: str | None,
        content_block_id: str | None,
        source_hash: str | None,
        source_structure_version: int | None,
    ) -> ResolvedReadingTarget:
        if task_unit_id is None:
            raise ValueError("content_block target requires task_unit_id")
        if content_block_id is None:
            raise ValueError("content_block target requires content_block_id")

        task_unit, section, parent_chapter = self._find_unique_task_unit(
            document=document,
            task_unit_id=task_unit_id,
        )
        if section_id is not None and section.section_id != section_id:
            raise ValueError(
                "content_block parent section mismatch: "
                f"content_block_id='{content_block_id}' task_unit_id='{task_unit_id}' "
                f"parent_section_id='{section.section_id}' requested_section_id='{section_id}'"
            )
        if chapter_id is not None and parent_chapter.chapter_id != chapter_id:
            raise ValueError(
                "content_block parent chapter mismatch: "
                f"content_block_id='{content_block_id}' task_unit_id='{task_unit_id}' "
                f"parent_chapter_id='{parent_chapter.chapter_id}' requested_chapter_id='{chapter_id}'"
            )
        if task_unit.parent_section_id is not None and (
            task_unit.parent_section_id != section.section_id
        ):
            raise ValueError(
                "content_block task_unit hierarchy parent mismatch: "
                f"task_unit_id='{task_unit_id}' parent_section_id='{task_unit.parent_section_id}' "
                f"hierarchy_section_id='{section.section_id}'"
            )

        content_block = self._find_unique_content_block(
            task_unit=task_unit,
            section=section,
            chapter=parent_chapter,
            content_block_id=content_block_id,
        )
        self._validate_content_block_source_evidence(
            content_block=content_block,
            source_hash=source_hash,
            source_structure_version=source_structure_version,
        )
        return ResolvedReadingTarget(
            document_id=document.document_id,
            document_title=document.title,
            document_language=document.language,
            target_level="content_block",
            target_id=content_block.block_id,
            content=content_block.content,
            chapter_id=parent_chapter.chapter_id,
            section_id=section.section_id,
            task_unit_id=task_unit.unit_id,
            content_block_id=content_block.block_id,
            chapter=parent_chapter,
            section=section,
            task_unit=task_unit,
            content_block=content_block,
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

    @classmethod
    def _content_from_document(cls, document: StructuredDocument) -> str:
        return cls._join_content_parts(
            cls._content_from_chapter(chapter) for chapter in document.chapters
        )

    @classmethod
    def _content_from_chapter(cls, chapter: StructuredChapter) -> str:
        return cls._join_content_parts(
            cls._content_from_section(section) for section in chapter.sections
        )

    @classmethod
    def _content_from_section(cls, section: StructuredSection) -> str:
        task_unit_content = cls._join_content_parts(
            task_unit.content for task_unit in section.task_units
        )
        if task_unit_content or section.task_units:
            return task_unit_content
        return section.content.strip()

    @staticmethod
    def _join_content_parts(parts: Iterable[str]) -> str:
        return "\n\n".join(
            part.strip()
            for part in parts
            if part.strip()
        )

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

    def _find_unique_content_block(
        self,
        *,
        task_unit: TaskUnit,
        section: StructuredSection | None = None,
        chapter: StructuredChapter | None = None,
        content_block_id: str,
    ) -> TaskUnitContentBlock:
        matches = [
            content_block
            for content_block in self.content_blocks_for_target_validation(
                task_unit=task_unit,
                section=section,
                chapter=chapter,
            )
            if content_block.block_id == content_block_id
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate_content_block_id:{content_block_id}")
        if not matches:
            raise ValueError(
                f"content_block_id '{content_block_id}' not found in "
                f"task_unit '{task_unit.unit_id}'"
            )
        return matches[0]

    @classmethod
    def _build_segmented_content_blocks_for_target_validation(
        cls,
        *,
        task_unit: TaskUnit,
        section: StructuredSection | None,
        chapter: StructuredChapter | None,
    ) -> list[TaskUnitContentBlock]:
        """Mirror the task-unit content endpoint's segmented render block ids."""
        content_blocks = task_unit.segment_content_blocks()
        if not content_blocks:
            return []

        leading_titles = {
            cls._normalize_render_heading(value)
            for value in (
                section.title if section is not None else None,
                chapter.title if chapter is not None else None,
            )
            if value
        }
        leading_titles.discard("")

        adjusted_blocks: list[TaskUnitContentBlock] = []
        for index, content_block in enumerate(content_blocks):
            if index == 0 and leading_titles:
                content_block = cls._drop_leading_render_heading(
                    task_unit_id=task_unit.unit_id,
                    content_block=content_block,
                    leading_titles=leading_titles,
                )
                if content_block is None:
                    continue
            adjusted_blocks.append(content_block)

        reindexed_blocks = [
            replace(
                content_block,
                block_id=build_default_content_block_id(
                    task_unit_id=task_unit.unit_id,
                    block_index=index,
                ),
                metadata=cls._reindex_content_block_metadata(
                    metadata=content_block.metadata,
                    task_unit_id=task_unit.unit_id,
                    block_index=index,
                ),
            )
            for index, content_block in enumerate(adjusted_blocks)
        ]
        return cls._normalize_content_blocks_for_target_validation(reindexed_blocks)

    @classmethod
    def _drop_leading_render_heading(
        cls,
        *,
        task_unit_id: str,
        content_block: TaskUnitContentBlock,
        leading_titles: set[str],
    ) -> TaskUnitContentBlock | None:
        lines = content_block.content.splitlines(keepends=True)
        if len(lines) < 2:
            return content_block

        first_line = lines[0].strip()
        if cls._normalize_render_heading(first_line) not in leading_titles:
            return content_block

        trimmed_content = "".join(lines[1:]).strip()
        if not trimmed_content:
            return None

        metadata = None if content_block.metadata is None else dict(content_block.metadata)
        if metadata is not None and isinstance(metadata.get("quote_span_start"), int):
            metadata["quote_span_start"] = metadata["quote_span_start"] + len(lines[0])

        return replace(
            content_block,
            block_id=build_default_content_block_id(
                task_unit_id=task_unit_id,
                block_index=0,
            ),
            content=trimmed_content,
            metadata=metadata,
        )

    @staticmethod
    def _reindex_content_block_metadata(
        *,
        metadata: dict | None,
        task_unit_id: str,
        block_index: int,
    ) -> dict | None:
        if metadata is None:
            return None
        updated = dict(metadata)
        updated["content_block_id"] = build_default_content_block_id(
            task_unit_id=task_unit_id,
            block_index=block_index,
        )
        return updated

    @staticmethod
    def _normalize_render_heading(value: str) -> str:
        return " ".join(normalize_ocr_text(value).strip().casefold().split())

    @staticmethod
    def _normalize_content_blocks_for_target_validation(
        content_blocks: list[TaskUnitContentBlock],
    ) -> list[TaskUnitContentBlock]:
        normalized_blocks: list[TaskUnitContentBlock] = []
        for block in content_blocks:
            normalized_content = normalize_ocr_text(block.content)
            if normalized_content == block.content:
                normalized_blocks.append(block)
                continue
            metadata = dict(block.metadata or {})
            metadata["display_normalization"] = "ocr_whitespace_v1"
            normalized_blocks.append(
                replace(
                    block,
                    content=normalized_content,
                    metadata=metadata,
                )
            )
        return normalized_blocks

    @classmethod
    def _validate_content_block_source_evidence(
        cls,
        *,
        content_block: TaskUnitContentBlock,
        source_hash: str | None,
        source_structure_version: int | None,
    ) -> None:
        if source_hash is None and source_structure_version is None:
            return

        metadata = content_block.metadata if isinstance(content_block.metadata, dict) else {}
        block_label = content_block.block_id.strip() or "<unknown>"
        if source_hash is not None:
            block_source_hash = cls._normalize_metadata_text(metadata.get("source_hash"))
            if block_source_hash is None:
                raise ValueError(
                    "stale_content_block_target: source_hash unavailable for "
                    f"content_block_id='{block_label}'"
                )
            if block_source_hash != source_hash:
                raise ValueError(
                    "stale_content_block_target: source_hash mismatch for "
                    f"content_block_id='{block_label}'"
                )

        if source_structure_version is not None:
            block_structure_version = cls._normalize_metadata_int(
                metadata.get("source_structure_version")
            )
            if block_structure_version is None:
                raise ValueError(
                    "stale_content_block_target: source_structure_version unavailable for "
                    f"content_block_id='{block_label}'"
                )
            if block_structure_version != source_structure_version:
                raise ValueError(
                    "stale_content_block_target: source_structure_version mismatch for "
                    f"content_block_id='{block_label}'"
                )

    @staticmethod
    def _normalize_metadata_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_metadata_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
