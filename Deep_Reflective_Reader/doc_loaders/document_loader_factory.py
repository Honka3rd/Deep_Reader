from .text_document_loader import TextDocumentLoader
from .pdf_document_loader import PdfDocumentLoader


class DocumentLoaderFactory:
    """Pick a loader implementation based on document extension."""
    def __init__(self):
        """Initialize object state and injected dependencies.
"""
        self.txt_loader = TextDocumentLoader()
        self.pdf_loader = PdfDocumentLoader()

    def get(self, doc_name: str):
        # 👉 最簡單判斷（可以之後優化）
        """Return configured value from provider.

Args:
    doc_name: Logical document name and artifact namespace (without extension).

Returns:
    Loader implementation matching ``doc_name`` extension."""
        if doc_name.endswith(".pdf"):
            return self.pdf_loader
        return self.txt_loader
