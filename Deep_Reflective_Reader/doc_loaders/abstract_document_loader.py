from abc import ABC, abstractmethod


class AbstractDocumentLoader(ABC):
    """Interface for loading document text by logical name."""

    @abstractmethod
    def open_source(self, doc_name: str, *, mode=None):
        """Capture a source session without decoding, OCR or semantic processing."""
        raise NotImplementedError

    @abstractmethod
    def source_path(self, doc_name: str):
        """Resolve the trusted local source for read-only identity verification."""
        raise NotImplementedError

    @abstractmethod
    def load(self, doc_name: str) -> str:
        """Load persisted artifact and return parsed object/data.

Args:
    doc_name: Logical document name and artifact namespace (without extension).

Returns:
    Full document text for the requested logical document name."""
        pass
