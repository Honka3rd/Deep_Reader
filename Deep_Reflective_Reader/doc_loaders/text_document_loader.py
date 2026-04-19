from pathlib import Path

from .abstract_document_loader import AbstractDocumentLoader


class TextDocumentLoader(AbstractDocumentLoader):
    """Load plain-text document files from raw data directory."""
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
    doc_name: Logical document name; supports both ``name`` and ``name.txt``.

Returns:
    UTF-8 text content loaded from ``data/raw``."""
        normalized_name = (
            doc_name
            if doc_name.lower().endswith(".txt")
            else f"{doc_name}.txt"
        )
        file_path = self.base_dir / normalized_name

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
