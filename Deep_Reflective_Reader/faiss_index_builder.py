from typing import Dict, List

import faiss
import numpy as np
from llama_index.core.schema import BaseNode

from embedder import Embedder
from faiss_index_bundle import FaissIndexBundle
from node_record import NodeRecord
from llm_provider import LLMProvider
from standardized.question_standardizer import QuestionStandardizer
from parsed_document import ParsedDocument
from prompt_assembler import PromptAssembler
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator

class FaissIndexBuilder:
    embedder: Embedder
    batch_size: int

    def __init__(
            self,
            embedder: Embedder,
            llm_provider: LLMProvider,
            question_standardizer: QuestionStandardizer,
            prompt_assembler: PromptAssembler,
            relevance_evaluator: QuestionRelevanceEvaluator,
            batch_size: int = 64
    ):
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.prompt_assembler = prompt_assembler
        self.batch_size = batch_size
        self.question_standardizer = question_standardizer
        self.relevance_evaluator = relevance_evaluator

    def build_from_parsed_document(self, parsed: ParsedDocument) -> FaissIndexBundle:
        dimension: int = self.embedder.probe_vector_dimension()

        base_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        faiss_index: faiss.IndexIDMap = faiss.IndexIDMap(base_index)

        id_to_record: Dict[int, NodeRecord] = {}
        next_id: int = 1
        nodes = parsed.nodes
        total_nodes: int = len(nodes)

        for start in range(0, total_nodes, self.batch_size):
            end: int = min(start + self.batch_size, total_nodes)
            batch_nodes: List[BaseNode] = nodes[start:end]
            print(f"FaissIndexBuilder#build progress: {end}/{total_nodes}")
            batch_texts: List[str] = [node.text.strip() for node in batch_nodes]
            batch_vectors: List[List[float]] = self.embedder.get_text_embedding_batch(batch_texts)

            batch_ids: List[int] = []
            batch_records: Dict[int, NodeRecord] = {}

            for node in batch_nodes:
                node_id: int = next_id
                next_id += 1

                batch_ids.append(node_id)
                batch_records[node_id] = NodeRecord(node, node_id)

            vectors_np: np.ndarray = np.array(batch_vectors, dtype=np.float32)
            ids_np: np.ndarray = np.array(batch_ids, dtype=np.int64)

            faiss_index.add_with_ids(vectors_np, ids_np)
            id_to_record.update(batch_records)

            print(f"FaissIndexBuilder#build progress: {end}/{total_nodes}")

        print(f"FaissIndexBuilder#build done: total_nodes={len(id_to_record)}")

        return FaissIndexBundle(
            faiss_index=faiss_index,
            embedder=self.embedder,
            id_to_record=id_to_record,
            dimension=dimension,
            llm_provider=self.llm_provider,
            question_standardizer=self.question_standardizer,
            prompt_assembler=self.prompt_assembler,
            document_language=parsed.document_language,
            relevance_evaluator=self.relevance_evaluator,
        )