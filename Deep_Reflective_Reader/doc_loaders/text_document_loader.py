from pathlib import Path

from .abstract_document_loader import AbstractDocumentLoader


class TextDocumentLoader(AbstractDocumentLoader):
    base_dir: Path

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)

    def load(self, doc_name: str) -> str:
        file_path = self.base_dir / f"{doc_name}.txt"

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()