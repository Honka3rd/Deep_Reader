from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentProfile:
    topic: str
    summary: str
    document_language: str