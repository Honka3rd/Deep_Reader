from pathlib import Path

from pypdf import PdfReader

from .abstract_document_loader import AbstractDocumentLoader


class PdfDocumentLoader(AbstractDocumentLoader):
    """Load PDF document files and join extracted page text."""
    base_dir: Path

    def __init__(self, base_dir: str = "data/raw"):
        """Initialize object state and injected dependencies.

Args:
    base_dir: Base dir.
"""
        self.base_dir = Path(base_dir)

    def load(self, doc_name: str) -> str:
        """Load persisted artifact and return parsed object/data.

Args:
    doc_name: Logical document name; supports both ``name`` and ``name.pdf``.

Returns:
    Concatenated text extracted from all PDF pages."""
        normalized_name = (
            doc_name
            if doc_name.lower().endswith(".pdf")
            else f"{doc_name}.pdf"
        )
        file_path = self.base_dir / normalized_name

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = PdfReader(str(file_path))

        texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        return "\n".join(texts)
