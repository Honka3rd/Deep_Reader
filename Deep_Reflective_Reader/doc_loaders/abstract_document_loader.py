from abc import ABC, abstractmethod


class AbstractDocumentLoader(ABC):
    """Interface for loading document text by logical name."""

    @abstractmethod
    def load(self, doc_name: str) -> str:
        """Load persisted artifact and return parsed object/data.

Args:
    doc_name: Logical document name and artifact namespace (without extension).

Returns:
    Full document text for the requested logical document name."""
        pass
