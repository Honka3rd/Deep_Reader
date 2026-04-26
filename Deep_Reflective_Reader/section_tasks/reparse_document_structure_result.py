from dataclasses import dataclass


@dataclass(frozen=True)
class ReparseDocumentStructureResult:
    """Execution result for explicit document structure reparse action."""

    success: bool
    doc_name: str
    parser_mode: str
    structured_document_path: str | None
    error: str | None
    section_count: int | None = None

    @classmethod
    def ok(
        cls,
        *,
        doc_name: str,
        parser_mode: str,
        structured_document_path: str | None,
        section_count: int | None = None,
    ) -> "ReparseDocumentStructureResult":
        """Build success reparse result."""
        return cls(
            success=True,
            doc_name=doc_name,
            parser_mode=parser_mode,
            structured_document_path=structured_document_path,
            error=None,
            section_count=section_count,
        )

    @classmethod
    def fail(
        cls,
        *,
        doc_name: str,
        parser_mode: str,
        error: str,
        structured_document_path: str | None = None,
    ) -> "ReparseDocumentStructureResult":
        """Build failure reparse result with explicit reason."""
        normalized_error = error.strip() or "reparse_document_structure_failed"
        return cls(
            success=False,
            doc_name=doc_name,
            parser_mode=parser_mode,
            structured_document_path=structured_document_path,
            error=normalized_error,
            section_count=None,
        )
