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
    """Save/load FAISS index artifacts and serialized node records."""
    embedder: Embedder
    def __init__(
            self,
            embedder: Embedder,
            llm_provider: LLMProvider,
            question_standardizer: QuestionStandardizer,
            prompt_assembler: PromptAssembler,
            relevance_evaluator: QuestionRelevanceEvaluator,
    ) -> None:
        """Initialize object state and injected dependencies.

Args:
    embedder: Embedder.
    llm_provider: Llm provider.
    question_standardizer: Question standardizer.
    prompt_assembler: Prompt assembler.
    relevance_evaluator: Relevance evaluator."""
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
        """Persist artifact/data to storage.

Args:
    bundle: Active FaissIndexBundle used for lookup and metadata access.
    config: StorageConfig describing filesystem artifact paths."""
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
                    "chunk_index": record.chunk_index(),
                    "char_start": record.char_start(),
                    "char_end": record.char_end(),
                    "prev_node_id": record.prev_node_id(),
                    "next_node_id": record.next_node_id(),
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
        """Load persisted artifact and return parsed object/data.

Args:
    config: StorageConfig describing filesystem artifact paths.

Returns:
    Reconstructed ``FaissIndexBundle`` loaded from persisted files."""
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
        """Remove persisted artifacts for clean rebuild.

Args:
    config: StorageConfig describing filesystem artifact paths."""
        index_path = config.get_raw_index_path()
        if os.path.exists(index_path):
            os.remove(index_path)
        records_path = config.get_raw_records_path()
        if os.path.exists(records_path):
            os.remove(records_path)

        print("FaissIndexStore#clear: index and records removed.")

    @staticmethod
    def has_position_metadata(config: StorageConfig) -> bool:
        """Check whether persisted records include required positional fields.

Args:
    config: StorageConfig describing filesystem artifact paths.

Returns:
    ``True`` when required positional fields exist in stored records."""
        records_path = config.get_raw_records_path()
        if not os.path.exists(records_path):
            return False

        try:
            with open(records_path, "r", encoding="utf-8") as f:
                payload: dict = json.load(f)

            records = payload.get("records")
            if not isinstance(records, dict) or not records:
                return False

            first_record = next(iter(records.values()))
            if not isinstance(first_record, dict):
                return False

            required_fields = {
                "chunk_index",
                "char_start",
                "char_end",
                "prev_node_id",
                "next_node_id",
            }
            return required_fields.issubset(first_record.keys())
        except Exception as e:
            print(f"FaissIndexStore#has_position_metadata failed: {e}")
            return False
