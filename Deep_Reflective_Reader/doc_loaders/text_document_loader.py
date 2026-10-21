from pathlib import Path
import hashlib

from .extraction_identity import ExtractionIdentity, ExtractionResult

from .abstract_document_loader import AbstractDocumentLoader
from .raw_data_paths import resolve_raw_data_dir


class TextDocumentLoader(AbstractDocumentLoader):
    """Load plain-text document files from raw data directory."""
    base_dir: Path

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize object state and injected dependencies.

Args:
    base_dir: Base dir.
"""
        self.base_dir = resolve_raw_data_dir(base_dir)

    def load(self, doc_name: str) -> str:
        """Load persisted artifact and return parsed object/data.

Args:
    doc_name: Logical document name; supports both ``name`` and ``name.txt``.

Returns:
    UTF-8 text content loaded from ``data/raw``."""
        return self.extract(doc_name).canonical_text

    def source_path(self, doc_name: str) -> Path:
        normalized_name = (
            doc_name
            if doc_name.lower().endswith(".txt")
            else f"{doc_name}.txt"
        )
        file_path = (self.base_dir / normalized_name).resolve()
        if file_path.parent != self.base_dir.resolve():
            raise ValueError("source_identity_invalid")

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return file_path

    def extract(self, doc_name: str, *, source_bytes: bytes | None = None) -> ExtractionResult:
        source = self.source_path(doc_name).read_bytes() if source_bytes is None else source_bytes
        identity = self._identity(source)
        text = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return ExtractionResult(identity, text)

    @staticmethod
    def _identity(source):
        return ExtractionIdentity.create(
            source_sha256=hashlib.sha256(source).hexdigest(), source_format="txt",
            extractor_id="text-loader", extraction_version="utf8-native-v1", mode="native",
            settings={"encoding": "utf-8", "errors": "strict", "bom": "preserve", "newlines": "universal"},
        )

    def open_source(self, doc_name: str, *, mode=None):
        from .source_session import SourceSession, SourceArtifact
        from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason
        if mode not in (None, "fast", "native"):
            raise PdfExtractionError(PdfExtractionFailureReason.MODE_NOT_APPLICABLE,
                                     "OCR and auto extraction modes apply only to PDF sources.")
        path = self.source_path(doc_name)
        source = path.read_bytes()
        return SourceSession(identity=self._identity(source), source_path=path,
            extract=lambda: SourceArtifact(self.extract(doc_name, source_bytes=source)),
            status=lambda state: {"requested_pdf_extraction_mode": mode,
                                  "pdf_extraction_provenance": "not_applicable"})
