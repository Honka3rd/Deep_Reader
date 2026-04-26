from enum import StrEnum

from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.section_splitter import CommonSectionSplitter
from language.language_code import LanguageCode


class SectionSplitterMode(StrEnum):
    """Selectable parser mode for section splitting."""

    COMMON = "common"
    LLM_ENHANCED = "llm_enhanced"


class SectionSplitterSelector:
    """Resolve splitter implementation by configured parser mode."""

    def __init__(
        self,
        common_splitter: CommonSectionSplitter | None = None,
        llm_splitter: LLMSectionSplitter | None = None,
    ):
        self.common_splitter = common_splitter or CommonSectionSplitter()
        self.llm_splitter = llm_splitter or LLMSectionSplitter(
            common_splitter=self.common_splitter
        )

    def get_splitter(
        self,
        mode: SectionSplitterMode = SectionSplitterMode.COMMON,
    ) -> AbstractSectionSplitter:
        """Return splitter implementation for target mode."""
        if mode == SectionSplitterMode.LLM_ENHANCED:
            return self.llm_splitter
        return self.common_splitter

    def split(
        self,
        *,
        raw_text: str,
        language: LanguageCode,
        mode: SectionSplitterMode = SectionSplitterMode.COMMON,
    ):
        """Convenience method: split by selected mode."""
        splitter = self.get_splitter(mode=mode)
        return splitter.split(raw_text=raw_text, language=language)
