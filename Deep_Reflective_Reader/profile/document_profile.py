from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentProfile:
    """Document profile data used for prompt conditioning."""
    topic: str
    summary: str
    document_language: str