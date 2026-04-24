from dataclasses import dataclass
from enum import StrEnum

from document_structure.structured_document import StructuredDocument, StructuredSection


class SectionTaskContextReason(StrEnum):
    """Soft-quality reasons for task context degradation."""

    UNKNOWN_DOCUMENT = "Unknown Document"
    UNTITLED_SECTION = "Untitled Section"
    NO_CONTAINER_TITLE = "None"



@dataclass(frozen=True)
class SectionTaskContext:
    """Normalized chapter-task context shared by section-based task services."""

    document_title: str
    container_title: str
    section_title: str
    section_id: str
    section_index: int
    section_content: str
    valid: bool
    reason: SectionTaskContextReason | None


class SectionTaskContextBuilder:
    """Build validated task context from structured document + section inputs."""

    def build_from_document(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> SectionTaskContext:
        """Resolve one section in document and build validated task context."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")

        target_section = self._find_section_by_id(
            document=document,
            section_id=normalized_section_id,
        )
        if target_section is None:
            raise ValueError(
                f"section_id '{normalized_section_id}' not found in document '{document.document_id}'"
            )

        return self.build_from_section(
            section=target_section,
            document_title=document.title,
        )

    @staticmethod
    def build_from_section(
        *,
        section: StructuredSection,
        document_title: str | None = None,
    ) -> SectionTaskContext:
        """Build validated task context directly from one structured section."""
        content = section.content.strip()
        if not content:
            raise ValueError(
                f"section '{section.section_id}' has empty content and cannot build task context"
            )

        resolved_document_title = (document_title or "").strip()
        resolved_section_title = (section.title or "").strip()
        resolved_container_title = (section.container_title or "").strip()
        has_unknown_document = not resolved_document_title
        has_untitled_section = not resolved_section_title
        has_no_container_title = not resolved_container_title

        valid = not (has_unknown_document or has_untitled_section or has_no_container_title)
        reason: SectionTaskContextReason | None = None
        if has_unknown_document:
            reason = SectionTaskContextReason.UNKNOWN_DOCUMENT
        elif has_untitled_section:
            reason = SectionTaskContextReason.UNTITLED_SECTION
        elif has_no_container_title:
            reason = SectionTaskContextReason.NO_CONTAINER_TITLE

        return SectionTaskContext(
            document_title=(
                resolved_document_title
            ),
            container_title=(
                resolved_container_title
            ),
            section_title=(
                resolved_section_title
            ),
            section_id=section.section_id,
            section_index=section.section_index,
            section_content=section.content,
            valid=valid,
            reason=reason,
        )

    @staticmethod
    def _find_section_by_id(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> StructuredSection | None:
        """Find one section by stable section id inside structured document."""
        for section in document.sections:
            if section.section_id == section_id:
                return section
        return None
