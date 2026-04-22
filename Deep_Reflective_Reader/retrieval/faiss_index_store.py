import json
from typing import Dict
import os

import faiss

from embeddings.embedder import Embedder
from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.node_record import NodeRecord
from config.faiss_storage_config import FaissStorageConfig
from llm.llm_provider import LLMProvider
from llm.llm_model_capabilities import LLMModelCapabilities
from config.token_budget_resolver import EffectiveTokenBudgets, resolve_effective_token_budgets

class FaissIndexStore:
    """Save/load FAISS index artifacts and serialized node records."""
    embedder: Embedder
    llm_provider: LLMProvider
    model_capabilities: LLMModelCapabilities | None
    token_budgets: EffectiveTokenBudgets

    def __init__(
            self,
            embedder: Embedder,
            llm_provider: LLMProvider,
            *,
            target_max_input_tokens: int,
            target_max_output_tokens: int,
            target_max_context_tokens: int,
            input_budget_utilization_ratio: float,
            context_budget_utilization_ratio: float,
    ) -> None:
        """Initialize object state and injected dependencies.

Args:
    embedder: Embedder.
    llm_provider: Llm provider.
    target_max_input_tokens: App input-token target before model-capability clamp.
    target_max_output_tokens: App output-token target before model-capability clamp.
    target_max_context_tokens: App context-token target for retrieval context size.
    input_budget_utilization_ratio: Safe utilization ratio for model input capacity.
    context_budget_utilization_ratio: Safe utilization ratio for effective input -> context."""
        self.embedder = embedder
        self.llm_provider = llm_provider
        capabilities: LLMModelCapabilities | None
        try:
            capabilities = self.llm_provider.get_model_capabilities()
        except Exception as e:
            print(
                "Warn:TokenBudget#store: failed to read model capabilities, "
                f"using fallback policy. error={e}"
            )
            capabilities = None
        self.model_capabilities = capabilities

        self.token_budgets = resolve_effective_token_budgets(
            capabilities=capabilities,
            target_max_input_tokens=target_max_input_tokens,
            target_max_output_tokens=target_max_output_tokens,
            target_max_context_tokens=target_max_context_tokens,
            input_budget_utilization_ratio=input_budget_utilization_ratio,
            context_budget_utilization_ratio=context_budget_utilization_ratio,
        )
        print(
            "TokenBudget#store:",
            f"model_capability=model={self.token_budgets.capability_model_name},"
            f"endpoint={self.token_budgets.capability_endpoint_kind},"
            f"max_input={self.token_budgets.capability_max_input_tokens},"
            f"max_output={self.token_budgets.capability_max_output_tokens}",
            f"utilization_ratios=input={self.token_budgets.input_budget_utilization_ratio},"
            f"context={self.token_budgets.context_budget_utilization_ratio}",
            f"effective_input_budget={self.token_budgets.effective_input_budget}",
            f"effective_output_budget={self.token_budgets.effective_output_budget}",
            f"effective_context_budget={self.token_budgets.effective_context_budget}",
            f"fallback_used={self.token_budgets.fallback_used}",
        )

    @staticmethod
    def save(
            bundle: FaissIndexBundle,
            config: FaissStorageConfig
    ) -> None:
        """Persist artifact/data to storage.

Args:
    bundle: Active FaissIndexBundle used for lookup and metadata access.
    config: FaissStorageConfig describing filesystem artifact paths."""
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
        config: FaissStorageConfig
    ) -> FaissIndexBundle:
        """Load persisted artifact and return parsed object/data.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

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
            model_capabilities=self.model_capabilities,
            id_to_record=id_to_record,
            dimension=payload["dimension"],
            document_language=payload["document_language"],
            max_context_tokens=self.token_budgets.effective_context_budget,
            max_prompt_tokens=self.token_budgets.effective_input_budget,
            reserved_output_tokens=self.token_budgets.effective_output_budget,
        )

    @staticmethod
    def clear(
            config: FaissStorageConfig
    ) -> None:
        """Remove persisted artifacts for clean rebuild.

Args:
    config: FaissStorageConfig describing filesystem artifact paths."""
        index_path = config.get_raw_index_path()
        if os.path.exists(index_path):
            os.remove(index_path)
        records_path = config.get_raw_records_path()
        if os.path.exists(records_path):
            os.remove(records_path)

        print("FaissIndexStore#clear: index and records removed.")

    @staticmethod
    def has_position_metadata(config: FaissStorageConfig) -> bool:
        """Check whether persisted records include required positional fields.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

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
