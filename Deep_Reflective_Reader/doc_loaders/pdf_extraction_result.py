"""Canonical extraction output and mode-bound optional evidence providers."""

from collections.abc import Callable
from dataclasses import dataclass, field

from .pdf_extraction_contract import PdfExtractionMode
from .pdf_outline import PdfOutlineResult
from .pdf_page_evidence import PdfPageBoundaryEvidenceResult, PdfPageLayoutEvidence


@dataclass(frozen=True)
class PdfExtractionResult:
    text: str
    pages: tuple[str, ...]
    boundary_evidence: PdfPageBoundaryEvidenceResult
    effective_mode: PdfExtractionMode
    extraction_version: str
    empty_page_indexes: tuple[int, ...]
    warnings: tuple[str, ...]
    layout_provider: Callable[[], list[PdfPageLayoutEvidence]]
    outline_provider: Callable[[], PdfOutlineResult]
    provenance: dict[str, object] = field(default_factory=dict)
    page_quality: tuple[dict[str, object], ...] = ()

    def as_extraction_result(self, identity):
        """Adapt the already extracted text; do not invoke PDF engines again."""
        from .extraction_identity import ExtractionResult, normalize_pdf_identity
        normalized = normalize_pdf_identity(identity)
        if (normalized.source_sha256 != self.boundary_evidence.source_sha256
                or normalized.mode != self.effective_mode
                or normalized.extraction_version != self.extraction_version):
            raise ValueError("pdf_extraction_identity_mismatch")
        return ExtractionResult(normalized, self.text)
