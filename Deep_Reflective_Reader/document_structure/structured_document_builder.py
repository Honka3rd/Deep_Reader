from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.structured_document import StructuredDocument, StructuredSection
from language.language_code import LanguageCode


class StructuredDocumentBuilder:
    """Build one StructuredDocument from metadata + raw text + language."""

    def __init__(
        self,
        section_splitter: AbstractSectionSplitter | None = None,
    ):
        """Initialize builder with injected splitter dependency."""
        self.section_splitter = section_splitter or CommonSectionSplitter()

    def build(
        self,
        document_id: str,
        title: str,
        raw_text: str,
        language: LanguageCode,
        source_path: str | None = None,
    ) -> StructuredDocument:
        """Build a structured document with fallback on split errors."""
        try:
            sections = self.section_splitter.split(
                raw_text=raw_text,
                language=language,
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
                )

            return StructuredDocument(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language.value,
                raw_text=raw_text,
                sections=sections,
                structure_error_code=None,
                structure_error_message=None,
            )
        except ValueError as error:
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code=self._map_value_error_code(error),
                error_message=str(error),
            )
        except Exception as error:
            return self._build_fallback_document(
                document_id=document_id,
                title=title,
                source_path=source_path,
                language=language,
                raw_text=raw_text,
                error_code="section_split_unexpected_error",
                error_message=str(error),
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
        )
        return StructuredDocument(
            document_id=document_id,
            title=title,
            source_path=source_path,
            language=language.value,
            raw_text=raw_text,
            sections=[fallback_section],
            structure_error_code=error_code,
            structure_error_message=error_message,
        )
