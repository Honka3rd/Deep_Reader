from llama_index.core.schema import BaseNode
from typing import List
class ParsedDocument:
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
        self.nodes = nodes
        self.document_language = document_language
        self.source = source
        self.title = title
        self.chapter = chapter