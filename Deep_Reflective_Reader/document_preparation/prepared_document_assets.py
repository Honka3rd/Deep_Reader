from dataclasses import dataclass, field
from typing import Any


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
        )
