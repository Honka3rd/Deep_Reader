from pathlib import Path

from .text_document_loader import TextDocumentLoader
from .pdf_document_loader import PdfDocumentLoader


class DocumentLoaderFactory:
    """Pick a loader implementation based on document extension."""
    def __init__(self):
        """Initialize object state and injected dependencies.
"""
        self.txt_loader = TextDocumentLoader()
        self.pdf_loader = PdfDocumentLoader()
        self.base_dir = Path("data/raw")

    def get(self, doc_name: str):
        # 👉 最簡單判斷（可以之後優化）
        """Return configured value from provider.

Args:
    doc_name: Logical document name; supports both with and without extension.

Returns:
    Loader implementation matching ``doc_name`` extension."""
        lowered_name = doc_name.lower()
        if lowered_name.endswith(".pdf"):
            return self.pdf_loader
        if lowered_name.endswith(".txt"):
            return self.txt_loader

        pdf_path = self.base_dir / f"{doc_name}.pdf"
        txt_path = self.base_dir / f"{doc_name}.txt"

        if pdf_path.exists() and not txt_path.exists():
            return self.pdf_loader
        if txt_path.exists() and not pdf_path.exists():
            return self.txt_loader

        # Keep historical default when no explicit extension / both-missing / both-exist.
        return self.txt_loader
