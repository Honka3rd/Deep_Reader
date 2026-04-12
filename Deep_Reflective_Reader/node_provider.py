from typing import List

from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from language.document_language_detector import DocumentLanguageDetector
from parsed_document import ParsedDocument
from storage_config import StorageConfig

class NodeProvider(object):
    """Parse raw document text into retrievable nodes with positional metadata."""
    parser: SentenceSplitter
    language_detector: DocumentLanguageDetector
    def __init__(
            self,
            parser: SentenceSplitter,
            detector: DocumentLanguageDetector
    ):
        # Text chunking behavior can be overridden by injecting a custom parser.
        """Initialize object state and injected dependencies.

Args:
    parser: Parser.
    detector: Detector.
"""
        self.parser = parser
        self.language_detector = detector

    def parse(self, text: str, config: StorageConfig) -> ParsedDocument:
        """Parse document text into nodes and attach positional metadata.

Args:
    text: Input text content.
    config: StorageConfig describing filesystem artifact paths.

Returns:
    Parsed document with chunk nodes and detected document language."""
        document: Document = Document(text=text)
        document_language: str = self.language_detector.detect(
            text=text,
            config=config,
        )
        nodes: List[BaseNode] = self.parser.get_nodes_from_documents([document])

        # Attach stable positional metadata for downstream persistence/use.
        search_start: int = 0
        for i, node in enumerate(nodes):
            # IMPORTANT: mutate node.metadata directly; assigning then mutating a local dict
            # can be lost with pydantic-backed models.
            if getattr(node, "metadata", None) is None:
                node.metadata = {}

            node.metadata["chunk_index"] = i

            start_char_idx = getattr(node, "start_char_idx", None)
            end_char_idx = getattr(node, "end_char_idx", None)

            # Fallback if parser does not provide char offsets.
            if not isinstance(start_char_idx, int) or not isinstance(end_char_idx, int):
                node_text: str = (getattr(node, "text", "") or "").strip()
                if node_text:
                    fallback_start = text.find(node_text, search_start)
                    if fallback_start < 0:
                        fallback_start = text.find(node_text)

                    if fallback_start >= 0:
                        start_char_idx = fallback_start
                        end_char_idx = fallback_start + len(node_text)
                        # Keep overlap-friendly advancing strategy.
                        search_start = fallback_start + 1
            else:
                search_start = start_char_idx + 1

            node.metadata["char_start"] = start_char_idx if isinstance(start_char_idx, int) else None
            node.metadata["char_end"] = end_char_idx if isinstance(end_char_idx, int) else None

            prev_node = nodes[i - 1] if i > 0 else None
            next_node = nodes[i + 1] if i < len(nodes) - 1 else None

            if prev_node is not None:
                node.metadata["prev_node_id"] = (
                    getattr(prev_node, "node_id", None)
                    or getattr(prev_node, "id_", None)
                )
            else:
                node.metadata["prev_node_id"] = None

            if next_node is not None:
                node.metadata["next_node_id"] = (
                    getattr(next_node, "node_id", None)
                    or getattr(next_node, "id_", None)
                )
            else:
                node.metadata["next_node_id"] = None

        print(
            f"NodeProvider#parse: total_nodes={len(nodes)}, "
            f"document_language={document_language}"
        )
        return ParsedDocument(
            nodes=nodes,
            document_language=document_language,
        )
