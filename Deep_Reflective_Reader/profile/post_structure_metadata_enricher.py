from __future__ import annotations

from dataclasses import replace

from document_structure.structured_document import StructuredDocument
from profile.document_profile import (
    DocumentProfile,
    DocumentStructureShape,
    LikelihoodLevel,
    PostStructureMetadata,
)


class PostStructureMetadataEnricher:
    """Compute hierarchy-grounded profile metadata after structured parsing."""

    def enrich(
        self,
        *,
        profile: DocumentProfile,
        structured_document: StructuredDocument,
    ) -> DocumentProfile:
        metadata = self._build_metadata(structured_document=structured_document)
        return replace(profile, post_structure_metadata=metadata)

    def _build_metadata(
        self,
        *,
        structured_document: StructuredDocument,
    ) -> PostStructureMetadata:
        chapters = list(structured_document.chapters or [])
        if not chapters:
            return PostStructureMetadata(
                chapter_count=0,
                section_count=0,
                task_unit_count=0,
                title_uniqueness_risk=LikelihoodLevel.UNKNOWN,
                actual_structure_shape=DocumentStructureShape.UNKNOWN,
                notes=["missing_chapters_hierarchy"],
            )

        chapter_count = len(chapters)
        section_count = 0
        task_unit_count = 0
        implicit_section_count = 0
        explicit_section_count = 0

        front_matter_chapter_count = 0
        main_body_chapter_count = 0
        appendix_chapter_count = 0
        back_matter_chapter_count = 0
        unknown_region_chapter_count = 0

        chapter_titles: list[str] = []
        section_titles: list[str] = []
        section_char_lengths: list[int] = []
        notes: list[str] = []

        for chapter in chapters:
            chapter_role = (chapter.chapter_role or "").strip().lower()
            if chapter_role == "front_matter":
                front_matter_chapter_count += 1
            elif chapter_role in {"main_body", ""}:
                main_body_chapter_count += 1
            elif chapter_role == "appendix":
                appendix_chapter_count += 1
            elif chapter_role == "back_matter":
                back_matter_chapter_count += 1
            else:
                unknown_region_chapter_count += 1

            chapter_title = (chapter.title or "").strip()
            if chapter_title:
                chapter_titles.append(chapter_title)

            for section in chapter.sections:
                section_count += 1
                task_unit_count += len(section.task_units)

                section_title = (section.title or "").strip()
                if section_title:
                    section_titles.append(section_title)

                if section.is_implicit_section:
                    implicit_section_count += 1
                else:
                    explicit_section_count += 1

                content = section.content or ""
                section_char_lengths.append(len(content))
                if not content.strip():
                    notes.append(f"section_content_missing:{section.section_id}")

        duplicate_chapter_titles = self._duplicate_titles(chapter_titles)
        duplicate_section_titles = self._duplicate_titles(section_titles)
        repeated_local_chapter_titles = list(duplicate_chapter_titles)

        titled_nodes = len(chapter_titles) + len(section_titles)
        total_nodes = chapter_count + section_count
        title_coverage = (
            None if total_nodes == 0 else round(titled_nodes / total_nodes, 4)
        )
        avg_sections_per_chapter = (
            None if chapter_count == 0 else round(section_count / chapter_count, 4)
        )
        avg_task_units_per_section = (
            None if section_count == 0 else round(task_unit_count / section_count, 4)
        )
        max_section_char_length = (
            None if not section_char_lengths else max(section_char_lengths)
        )
        avg_section_char_length = (
            None
            if not section_char_lengths
            else round(sum(section_char_lengths) / len(section_char_lengths), 2)
        )

        title_uniqueness_risk = self._derive_title_uniqueness_risk(
            duplicate_chapter_titles=duplicate_chapter_titles,
            duplicate_section_titles=duplicate_section_titles,
            repeated_local_chapter_titles=repeated_local_chapter_titles,
            title_coverage=title_coverage,
            chapter_count=chapter_count,
            section_count=section_count,
        )
        actual_structure_shape, shape_notes = self._derive_actual_structure_shape(
            chapter_count=chapter_count,
            section_count=section_count,
            main_body_chapter_count=main_body_chapter_count,
            implicit_section_count=implicit_section_count,
            avg_sections_per_chapter=avg_sections_per_chapter,
            avg_section_char_length=avg_section_char_length,
            max_section_char_length=max_section_char_length,
            duplicate_chapter_titles=duplicate_chapter_titles,
            title_coverage=title_coverage,
        )
        notes.extend(shape_notes)
        notes = sorted(set(note for note in notes if note))

        return PostStructureMetadata(
            chapter_count=chapter_count,
            section_count=section_count,
            task_unit_count=task_unit_count,
            front_matter_chapter_count=front_matter_chapter_count,
            main_body_chapter_count=main_body_chapter_count,
            appendix_chapter_count=appendix_chapter_count,
            back_matter_chapter_count=back_matter_chapter_count,
            unknown_region_chapter_count=unknown_region_chapter_count,
            implicit_section_count=implicit_section_count,
            explicit_section_count=explicit_section_count,
            duplicate_chapter_titles=duplicate_chapter_titles,
            duplicate_section_titles=duplicate_section_titles,
            repeated_local_chapter_titles=repeated_local_chapter_titles,
            title_uniqueness_risk=title_uniqueness_risk,
            actual_structure_shape=actual_structure_shape,
            title_coverage=title_coverage,
            avg_sections_per_chapter=avg_sections_per_chapter,
            avg_task_units_per_section=avg_task_units_per_section,
            max_section_char_length=max_section_char_length,
            avg_section_char_length=avg_section_char_length,
            notes=notes,
        )

    def _duplicate_titles(self, titles: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        display: dict[str, str] = {}
        for title in titles:
            normalized = title.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, normalized)
        duplicates = [display[key] for key, count in counts.items() if count > 1]
        duplicates.sort(key=str.casefold)
        return duplicates

    def _derive_title_uniqueness_risk(
        self,
        *,
        duplicate_chapter_titles: list[str],
        duplicate_section_titles: list[str],
        repeated_local_chapter_titles: list[str],
        title_coverage: float | None,
        chapter_count: int,
        section_count: int,
    ) -> LikelihoodLevel:
        if chapter_count == 0 and section_count == 0:
            return LikelihoodLevel.UNKNOWN
        if repeated_local_chapter_titles or duplicate_chapter_titles:
            return LikelihoodLevel.HIGH
        if len(duplicate_section_titles) >= 3:
            return LikelihoodLevel.HIGH
        if duplicate_section_titles:
            return LikelihoodLevel.MEDIUM
        if title_coverage is None:
            return LikelihoodLevel.UNKNOWN
        if title_coverage < 0.4:
            return LikelihoodLevel.MEDIUM
        if title_coverage < 0.65:
            return LikelihoodLevel.LOW
        return LikelihoodLevel.NONE

    def _derive_actual_structure_shape(
        self,
        *,
        chapter_count: int,
        section_count: int,
        main_body_chapter_count: int,
        implicit_section_count: int,
        avg_sections_per_chapter: float | None,
        avg_section_char_length: float | None,
        max_section_char_length: int | None,
        duplicate_chapter_titles: list[str],
        title_coverage: float | None,
    ) -> tuple[DocumentStructureShape, list[str]]:
        notes: list[str] = []
        if chapter_count <= 0 or section_count <= 0:
            return DocumentStructureShape.UNKNOWN, notes

        if chapter_count <= 1 and section_count <= 1:
            if (max_section_char_length or 0) >= 6000:
                return DocumentStructureShape.FLAT_LONG_TEXT, notes
            return DocumentStructureShape.ESSAY_SECTIONS, notes

        if (
            section_count > 100
            and avg_section_char_length is not None
            and avg_section_char_length < 500
        ):
            return DocumentStructureShape.OVER_FRAGMENTED, notes

        if (
            main_body_chapter_count >= 5
            and avg_sections_per_chapter is not None
            and avg_sections_per_chapter <= 1.2
            and implicit_section_count >= max(3, int(section_count * 0.6))
        ):
            return DocumentStructureShape.CHAPTER_ONLY, notes

        if avg_sections_per_chapter is not None and avg_sections_per_chapter >= 2.0:
            return DocumentStructureShape.CHAPTER_SECTION, notes

        if (
            chapter_count <= 5
            and section_count <= 8
            and (title_coverage or 0.0) >= 0.5
        ):
            return DocumentStructureShape.ESSAY_SECTIONS, notes

        if duplicate_chapter_titles and main_body_chapter_count >= 4:
            notes.append("possible_part_chapter_repeated_local_titles")
            return DocumentStructureShape.PART_CHAPTER, notes

        return DocumentStructureShape.MIXED, notes

