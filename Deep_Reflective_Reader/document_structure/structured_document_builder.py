from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_selector import (
    SectionSplitterMode,
    SectionSplitterSelector,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_hierarchy_builder import (
    build_document_hierarchy_from_sections,
)
from language.language_code import LanguageCode


class StructuredDocumentBuilder:
    """Build one StructuredDocument from metadata + raw text + language."""

    def __init__(
        self,
        section_splitter: AbstractSectionSplitter | None = None,
        section_splitter_selector: SectionSplitterSelector | None = None,
    ):
        """Initialize builder with injected splitter dependency."""
        self.section_splitter = section_splitter or CommonSectionSplitter()
        self.section_splitter_selector = section_splitter_selector

    def build(
        self,
        document_id: str,
        title: str,
        raw_text: str,
        language: LanguageCode,
        source_path: str | None = None,
        parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
    ) -> StructuredDocument:
        """Build a structured document with fallback on split errors."""
        try:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            if self.section_splitter_selector is not None:
                sections, parse_provenance = (
                    self.section_splitter_selector.split_with_provenance(
                        raw_text=raw_text,
                        language=language,
                        mode=resolved_mode,
                    )
                )
            else:
                sections = self.section_splitter.split(
                    raw_text=raw_text,
                    language=language,
                )
                parse_provenance = self._build_parse_provenance(
                    requested_parser_mode=SectionSplitterMode.COMMON.value,
                    effective_parser_mode=SectionSplitterMode.COMMON.value,
                    fallback_used=False,
                    fallback_reason=None,
                    source="common_section_splitter",
                )
            if not sections:
                return self._build_fallback_document(
                    document_id=document_id,
                    title=title,
                    source_path=source_path,
                    language=language,
                    raw_text=raw_text,
                    error_code="empty_sections_result",
                    error_message="SectionSplitter returned no sections.",
                    parse_provenance=self._build_parse_provenance(
                        requested_parser_mode=resolved_mode.value,
                        effective_parser_mode="fallback_document",
                        fallback_used=True,
                        fallback_reason="empty_sections_result",
                        source="structured_document_builder",
                    ),
                )

            document = StructuredDocument(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language.value,
                raw_text=raw_text,
                sections=sections,
                structure_error_code=None,
                structure_error_message=None,
                parse_provenance=parse_provenance,
            )
            return build_document_hierarchy_from_sections(document)
        except ValueError as error:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code=self._map_value_error_code(error),
                error_message=str(error),
                parse_provenance=self._build_parse_provenance(
                    requested_parser_mode=resolved_mode.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason=self._map_value_error_code(error),
                    source="structured_document_builder",
                ),
            )
        except Exception as error:
            resolved_mode = SectionSplitterMode.resolve(parser_mode)
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code="section_split_unexpected_error",
                error_message=str(error),
                parse_provenance=self._build_parse_provenance(
                    requested_parser_mode=resolved_mode.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason="section_split_unexpected_error",
                    source="structured_document_builder",
                ),
            )

    @staticmethod
    def _map_value_error_code(error: ValueError) -> str:
        """Map expected ValueError messages to stable structure error codes."""
        message = str(error).lower()
        if "unsupported document structure language" in message:
            return "unsupported_structure_language"
        return "section_split_value_error"

    @staticmethod
    def _build_fallback_document(
        *,
        document_id: str,
        title: str,
        source_path: str | None,
        language: LanguageCode,
        raw_text: str,
        error_code: str,
        error_message: str,
        parse_provenance: dict[str, object] | None = None,
    ) -> StructuredDocument:
        """Build fallback document with one full-text section."""
        fallback_section = StructuredSection(
            section_id="section-0",
            section_index=0,
            title=None,
            level=1,
            content=raw_text,
            char_start=0,
            char_end=len(raw_text),
            section_role=SectionRole.MAIN_BODY,
        )
        fallback_document = StructuredDocument(
            document_id=document_id,
            title=title,
            source_path=source_path,
            language=language.value,
            raw_text=raw_text,
            sections=[fallback_section],
            structure_error_code=error_code,
            structure_error_message=error_message,
            parse_provenance=(
                dict(parse_provenance)
                if parse_provenance is not None
                else StructuredDocumentBuilder._build_parse_provenance(
                    requested_parser_mode=SectionSplitterMode.COMMON.value,
                    effective_parser_mode="fallback_document",
                    fallback_used=True,
                    fallback_reason=error_code,
                    source="structured_document_builder",
                )
            ),
        )
        return build_document_hierarchy_from_sections(fallback_document)

    @staticmethod
    def _build_parse_provenance(
        *,
        requested_parser_mode: str,
        effective_parser_mode: str,
        fallback_used: bool,
        fallback_reason: str | None,
        source: str,
    ) -> dict[str, object]:
        return {
            "requested_parser_mode": requested_parser_mode,
            "effective_parser_mode": effective_parser_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "source": source,
        }
