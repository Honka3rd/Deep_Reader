from typing import Dict, List

import faiss
import numpy as np
from llama_index.core.schema import BaseNode

from context.document_context_builder import DocumentContextBuilder
from context.token_budget_manager import TokenBudgetManager
from embeddings.embedder import Embedder
from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.node_record import NodeRecord
from llm.llm_provider import LLMProvider
from retrieval.parsed_document import ParsedDocument
from llm.llm_model_capabilities import LLMModelCapabilities
from config.token_budget_resolver import EffectiveTokenBudgets, resolve_effective_token_budgets

class FaissIndexBuilder:
    """Create FAISS index structures from parsed document nodes."""
    embedder: Embedder
    llm_provider: LLMProvider
    model_capabilities: LLMModelCapabilities | None
    batch_size: int
    token_budgets: EffectiveTokenBudgets
    token_budget_manager: TokenBudgetManager
    document_context_builder: DocumentContextBuilder

    def __init__(
            self,
            embedder: Embedder,
            llm_provider: LLMProvider,
            token_budget_manager: TokenBudgetManager,
            document_context_builder: DocumentContextBuilder,
            *,
            target_max_input_tokens: int,
            target_max_output_tokens: int,
            target_max_context_tokens: int,
            input_budget_utilization_ratio: float,
            context_budget_utilization_ratio: float,
            batch_size: int,
    ):
        """Initialize object state and injected dependencies.

Args:
    embedder: Embedder.
    llm_provider: Llm provider.
    token_budget_manager: Token-budget manager singleton.
    document_context_builder: Document-context builder singleton.
    target_max_input_tokens: App input-token target before model-capability clamp.
    target_max_output_tokens: App output-token target before model-capability clamp.
    target_max_context_tokens: App context-token target for retrieval context size.
    input_budget_utilization_ratio: Safe utilization ratio for model input capacity.
    context_budget_utilization_ratio: Safe utilization ratio for effective input -> context.
    batch_size: Batch size.
"""
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.token_budget_manager = token_budget_manager
        self.document_context_builder = document_context_builder
        self.batch_size = batch_size
        capabilities: LLMModelCapabilities | None
        try:
            capabilities = self.llm_provider.get_model_capabilities()
        except Exception as e:
            print(
                "Warn:TokenBudget#builder: failed to read model capabilities, "
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
            "TokenBudget#builder:",
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

    def build_from_parsed_document(self, parsed: ParsedDocument) -> FaissIndexBundle:
        """Build a FAISS index bundle from parsed nodes.

Args:
    parsed: Parsed.

Returns:
    Ready-to-query ``FaissIndexBundle`` built from parsed document nodes."""
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
            model_capabilities=self.model_capabilities,
            id_to_record=id_to_record,
            dimension=dimension,
            token_budget_manager=self.token_budget_manager,
            document_context_builder=self.document_context_builder,
            document_language=parsed.document_language,
            max_context_tokens=self.token_budgets.effective_context_budget,
            max_prompt_tokens=self.token_budgets.effective_input_budget,
            reserved_output_tokens=self.token_budgets.effective_output_budget,
        )
