"""Pure comparison of applied enhancement hierarchies on one canonical source."""

from dataclasses import dataclass
from enum import StrEnum

from document_structure.structured_document import StructuredDocument


class EnhancementChangeKind(StrEnum):
    UNCHANGED = "unchanged"
    TITLES_ONLY = "titles_only"
    STRUCTURE_CHANGED = "structure_changed"


class EnhancementComparisonError(ValueError):
    """Safe reason code only; never includes source text or navigation labels."""


@dataclass(frozen=True)
class NodeCorrespondence:
    node_type: str
    chapter_index: int
    section_index: int | None
    existing_id: str
    candidate_id: str


@dataclass(frozen=True)
class EnhancementChange:
    kind: EnhancementChangeKind
    correspondence: tuple[NodeCorrespondence, ...]
    existing_chapters: int
    existing_sections: int
    candidate_chapters: int
    candidate_sections: int

    def diagnostics(self) -> dict[str, str | int]:
        # These are applied/validated hierarchy counts, not raw provider proposals.
        return {
            "change_kind": self.kind.value,
            "existing_chapter_count": self.existing_chapters,
            "existing_section_count": self.existing_sections,
            "accepted_candidate_chapter_count": self.candidate_chapters,
            "accepted_candidate_section_count": self.candidate_sections,
        }


def _label(value: str | None) -> str:
    if value is not None and not isinstance(value, str):
        raise EnhancementComparisonError("invalid_label")
    return (value or "").strip()


def _signature(document: StructuredDocument, source: str) -> tuple[tuple, tuple]:
    if not document.chapters or document.structure_error_code:
        raise EnhancementComparisonError("invalid_hierarchy")
    # PostgreSQL readback may omit raw_text; sections must still cover exact source.
    if document.raw_text and document.raw_text != source:
        raise EnhancementComparisonError("canonical_source_mismatch")
    cursor = 0
    chapter_ids: set[str] = set()
    section_ids: set[str] = set()
    structure = []
    labels = []
    for chapter in document.chapters:
        if not chapter.chapter_id or chapter.chapter_id in chapter_ids or not chapter.sections:
            raise EnhancementComparisonError("invalid_chapter_identity")
        chapter_ids.add(chapter.chapter_id)
        if type(chapter.level) is not int or chapter.level < 1:
            raise EnhancementComparisonError("invalid_level")
        sections = []
        section_labels = []
        for section in chapter.sections:
            if not section.section_id or section.section_id in section_ids:
                raise EnhancementComparisonError("invalid_section_identity")
            section_ids.add(section.section_id)
            if section.parent_chapter_id not in (None, chapter.chapter_id):
                raise EnhancementComparisonError("invalid_parent")
            if type(section.level) is not int or section.level < 1:
                raise EnhancementComparisonError("invalid_level")
            start, end = section.char_start, section.char_end
            if (type(start) is not int or type(end) is not int
                    or start != cursor or end <= start or end > len(source)):
                raise EnhancementComparisonError("invalid_source_coverage")
            if section.content != source[start:end]:
                raise EnhancementComparisonError("canonical_source_mismatch")
            sections.append((start, end, section.level, section.section_role))
            section_labels.append((_label(section.title), _label(section.container_title)))
            cursor = end
        structure.append((chapter.level, chapter.chapter_role, tuple(sections)))
        labels.append((_label(chapter.title), tuple(section_labels)))
    if cursor != len(source):
        raise EnhancementComparisonError("incomplete_source_coverage")
    return tuple(structure), tuple(labels)


def compare_enhancement_documents(
    existing: StructuredDocument,
    candidate: StructuredDocument,
    *,
    canonical_text: str,
) -> EnhancementChange:
    """Validate both hierarchies; map old identities only when structure is equal.

    The caller supplies the accepted source snapshot, not candidate-generated text.
    No persistence, generation, artifact inspection or task-unit rebuilding occurs.
    """
    if not isinstance(canonical_text, str) or not canonical_text:
        raise EnhancementComparisonError("missing_canonical_source")
    if existing.document_id != candidate.document_id:
        raise EnhancementComparisonError("document_identity_mismatch")
    old_structure, old_labels = _signature(existing, canonical_text)
    new_structure, new_labels = _signature(candidate, canonical_text)
    correspondence = []
    if old_structure != new_structure:
        kind = EnhancementChangeKind.STRUCTURE_CHANGED
    else:
        kind = (EnhancementChangeKind.UNCHANGED if old_labels == new_labels
                else EnhancementChangeKind.TITLES_ONLY)
        for ci, (old_chapter, new_chapter) in enumerate(zip(existing.chapters, candidate.chapters)):
            correspondence.append(NodeCorrespondence(
                "chapter", ci, None, old_chapter.chapter_id, new_chapter.chapter_id))
            for si, (old_section, new_section) in enumerate(zip(old_chapter.sections, new_chapter.sections)):
                correspondence.append(NodeCorrespondence(
                    "section", ci, si, old_section.section_id, new_section.section_id))
    return EnhancementChange(
        kind=kind,
        correspondence=tuple(correspondence),
        existing_chapters=len(existing.chapters),
        existing_sections=sum(len(ch.sections) for ch in existing.chapters),
        candidate_chapters=len(candidate.chapters),
        candidate_sections=sum(len(ch.sections) for ch in candidate.chapters),
    )
