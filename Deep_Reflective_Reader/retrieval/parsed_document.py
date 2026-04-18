from llama_index.core.schema import BaseNode
from typing import List
class ParsedDocument:
    """Container for parsed nodes plus document-level parsing attributes."""
    nodes: List[BaseNode]
    document_language: str
    source: str | None = None
    title: str | None = None
    chapter: str | None = None
    def __init__(
            self,
            nodes: list[BaseNode],
            document_language: str | None = None,
            source: str | None = None,
            title: str | None = None,
            chapter: str | None = None
    ):
        """Initialize object state and injected dependencies.

Args:
    nodes: Nodes.
    document_language: Primary document language code (e.g. en/zh).
    source: Source.
    title: Title.
    chapter: Chapter.
"""
        self.nodes = nodes
        self.document_language = document_language
        self.source = source
        self.title = title
        self.chapter = chapter
