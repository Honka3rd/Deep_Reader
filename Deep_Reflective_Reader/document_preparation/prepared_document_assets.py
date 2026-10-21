from dataclasses import dataclass, field
from typing import Any
from document_structure.common_parse_policy import TocHint
from doc_loaders.pdf_extraction_contract import PdfExtractionMode, PdfExtractionProvenance
from .pdf_extraction_identity import PdfExtractionIdentity
from doc_loaders.extraction_identity import ExtractionResult


@dataclass
class PreparedDocumentAssets:
    """Preparation result snapshot for one document lifecycle."""

    doc_name: str
    raw_text: str | None
    language: str | None
    structured_document_ready: bool
    faiss_ready: bool
    profile_ready: bool
    bundle_ready: bool
    structured_document_path: str | None
    faiss_namespace: str | None
    errors: list[str] = field(default_factory=list)
    requested_pdf_extraction_mode: PdfExtractionMode | None = None
    effective_pdf_extraction_mode: PdfExtractionMode | None = None
    pdf_extraction_provenance: PdfExtractionProvenance = PdfExtractionProvenance.UNKNOWN
    pdf_extraction_identity: PdfExtractionIdentity | None = None
    requested_toc_hint: TocHint | None = None
    enhancement_intent: str = "restore_source"
    # Internal lifecycle authority; never accepted from serialized/API assets.
    profile_creation_allowed: bool = False
    profile_generated: bool = False
    extraction_result: ExtractionResult | None = None
    source_session: Any = field(default=None, repr=False)

    def apply_extraction_status(self, status):
        """Project legacy API metadata; never use it as extraction authority."""
        for key in ("requested_pdf_extraction_mode", "effective_pdf_extraction_mode",
                    "pdf_extraction_identity", "pdf_extraction_provenance"):
            if key in status:
                setattr(self, key, status[key])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dictionary payload."""
        return {
            "doc_name": self.doc_name,
            "raw_text": self.raw_text,
            "language": self.language,
            "structured_document_ready": self.structured_document_ready,
            "faiss_ready": self.faiss_ready,
            "profile_ready": self.profile_ready,
            "bundle_ready": self.bundle_ready,
            "structured_document_path": self.structured_document_path,
            "faiss_namespace": self.faiss_namespace,
            "errors": list(self.errors),
            "requested_toc_hint": self.requested_toc_hint,
            "enhancement_intent": self.enhancement_intent,
            "requested_pdf_extraction_mode": self.requested_pdf_extraction_mode,
            "effective_pdf_extraction_mode": self.effective_pdf_extraction_mode,
            "pdf_extraction_provenance": self.pdf_extraction_provenance,
            "pdf_extraction_identity": self.pdf_extraction_identity.to_dict() if self.pdf_extraction_identity else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedDocumentAssets":
        """Build DTO from dictionary payload."""
        return cls(
            doc_name=str(data["doc_name"]),
            raw_text=(None if data.get("raw_text") is None else str(data["raw_text"])),
            language=(None if data.get("language") is None else str(data["language"])),
            structured_document_ready=bool(data["structured_document_ready"]),
            faiss_ready=bool(data["faiss_ready"]),
            profile_ready=bool(data["profile_ready"]),
            bundle_ready=bool(data["bundle_ready"]),
            structured_document_path=(
                None
                if data.get("structured_document_path") is None
                else str(data["structured_document_path"])
            ),
            faiss_namespace=(
                None if data.get("faiss_namespace") is None else str(data["faiss_namespace"])
            ),
            errors=[str(error) for error in data.get("errors", [])],
            requested_toc_hint=TocHint(data["requested_toc_hint"]) if data.get("requested_toc_hint") is not None else None,
            enhancement_intent=str(data.get("enhancement_intent", "restore_source")),
            requested_pdf_extraction_mode=(
                PdfExtractionMode(data["requested_pdf_extraction_mode"])
                if data.get("requested_pdf_extraction_mode") is not None else None
            ),
            effective_pdf_extraction_mode=(
                PdfExtractionMode(data["effective_pdf_extraction_mode"])
                if data.get("effective_pdf_extraction_mode") is not None else None
            ),
            pdf_extraction_provenance=PdfExtractionProvenance(
                data.get("pdf_extraction_provenance", "unknown")
            ),
            pdf_extraction_identity=(PdfExtractionIdentity(**data["pdf_extraction_identity"])
                                     if data.get("pdf_extraction_identity") is not None else None),
        )
