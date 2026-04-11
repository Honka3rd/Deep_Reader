from pathlib import Path

from pypdf import PdfReader

from .abstract_document_loader import AbstractDocumentLoader


class PdfDocumentLoader(AbstractDocumentLoader):
    base_dir: Path

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)

    def load(self, doc_name: str) -> str:
        file_path = self.base_dir / f"{doc_name}.pdf"

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = PdfReader(str(file_path))

        texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        return "\n".join(texts)