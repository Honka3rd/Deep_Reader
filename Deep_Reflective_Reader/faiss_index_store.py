import json
from typing import Dict
import os

import faiss

from embedder import Embedder
from faiss_index_bundle import FaissIndexBundle
from node_record import NodeRecord
from storage_config import StorageConfig
from llm_provider import LLMProvider
from standardized.question_standardizer import QuestionStandardizer
from prompt_assembler import PromptAssembler
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator

class FaissIndexStore:
    embedder: Embedder
    def __init__(
            self,
            embedder: Embedder,
            llm_provider: LLMProvider,
            question_standardizer: QuestionStandardizer,
            prompt_assembler: PromptAssembler,
            relevance_evaluator: QuestionRelevanceEvaluator,
    ) -> None:
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.question_standardizer = question_standardizer
        self.prompt_assembler = prompt_assembler
        self.relevance_evaluator = relevance_evaluator

    @staticmethod
    def save(
            bundle: FaissIndexBundle,
            config: StorageConfig
    ) -> None:
        faiss.write_index(bundle.faiss_index, config.get_raw_index_path())

        payload: dict = {
            "dimension": bundle.dimension,
            "document_language": bundle.document_language,
            "records": {
                str(node_id): {
                    "node_id": record.node_id(),
                    "node_key": record.node_key(),
                    "text": record.text(),
                    "source": record.source(),
                    "chapter": record.chapter(),
                    "position": record.position(),
                }
                for node_id, record in bundle.id_to_record.items()
            }
        }

        with open(config.get_raw_records_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(
        self,
        config: StorageConfig
    ) -> FaissIndexBundle:
        loaded_index = faiss.read_index(config.get_raw_index_path())

        with open(config.get_raw_records_path(), "r", encoding="utf-8") as f:
            payload: dict = json.load(f)

        id_to_record: Dict[int, NodeRecord] = {
            int(node_id): NodeRecord.from_persisted_dict(record_dict)
            for node_id, record_dict in payload["records"].items()
        }
        return FaissIndexBundle(
            faiss_index=loaded_index,
            embedder=self.embedder,
            id_to_record=id_to_record,
            dimension=payload["dimension"],
            llm_provider=self.llm_provider,
            question_standardizer=self.question_standardizer,
            prompt_assembler=self.prompt_assembler,
            document_language=payload["document_language"],
            relevance_evaluator=self.relevance_evaluator
        )

    @staticmethod
    def clear(
            config: StorageConfig
    ) -> None:
        """
        刪除 faiss index 和 records，強制重建用
        """
        index_path = config.get_raw_index_path()
        if os.path.exists(index_path):
            os.remove(index_path)
        records_path = config.get_raw_records_path()
        if os.path.exists(records_path):
            os.remove(records_path)

        print("FaissIndexStore#clear: index and records removed.")