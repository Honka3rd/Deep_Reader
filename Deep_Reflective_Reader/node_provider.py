from typing import List

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from language.document_language_detector import DocumentLanguageDetector
from parsed_document import ParsedDocument
from storage_config import StorageConfig

class NodeProvider(object):
    parser: SentenceSplitter
    language_detector: DocumentLanguageDetector
    def __init__(
            self,
            parser: SentenceSplitter,
            detector: DocumentLanguageDetector
    ):
        # Text chunking behavior can be overridden by injecting a custom parser.
        self.parser = parser
        self.language_detector = detector

    def parse(self, text: str, config: StorageConfig) -> ParsedDocument:
        document: Document = Document(text=text)
        document_language: str = self.language_detector.detect(
            text=text,
            config=config,
        )
        nodes: List[BaseNode] = self.parser.get_nodes_from_documents([document])
        print(
            f"NodeProvider#parse: total_nodes={len(nodes)}, "
            f"document_language={document_language}"
        )
        return ParsedDocument(
            nodes=nodes,
            document_language=document_language,
        )
