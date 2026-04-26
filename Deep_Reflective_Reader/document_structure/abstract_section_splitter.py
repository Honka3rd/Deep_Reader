from abc import ABC, abstractmethod

from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode


class AbstractSectionSplitter(ABC):
    """Abstract splitter contract from raw text to flat structured sections."""

    @abstractmethod
    def split(
        self,
        raw_text: str,
        language: LanguageCode = LanguageCode.UNKNOWN,
    ) -> list[StructuredSection]:
        """Split raw text into structured sections."""
        raise NotImplementedError
