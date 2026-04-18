from dataclasses import replace

from app_DI_config import AppDIConfig
from container import ApplicationLookupContainer
from faiss_index_bundle import FaissIndexBundle
from qa_enums import AnswerLevel
from reading_session import ReadingSession
from session_manager import SessionUpdateResult

class Coordinator:
    """Coordinate document loading, index readiness, and session-aware QA orchestration."""
    chunk_size: int
    chunk_overlap: int
    embedding_model: str

    def __init__(
            self,
            chunk_size: int | None = None,
            chunk_overlap: int | None = None,
            embedding_model: str | None = None,
            llm_model: str | None = None,
            target_max_input_tokens: int | None = None,
            target_max_output_tokens: int | None = None,
            target_max_context_tokens: int | None = None,
            input_budget_utilization_ratio: float | None = None,
            context_budget_utilization_ratio: float | None = None,
            full_text_input_budget_utilization_ratio: float | None = None,
            full_text_context_budget_utilization_ratio: float | None = None,
            embedding_batch_size: int | None = None,
            bundle_cache_capacity: int | None = None,
            session_recent_limit: int | None = None,
            base_near_chunk_threshold: int | None = None,
            min_near_chunk_threshold: int | None = None,
            max_near_chunk_threshold: int | None = None,
            global_scope_min_top_k: int | None = None,
            global_coverage_chunk_gap: int | None = None,
    ):
        """Initialize runtime dependencies and in-memory session storage.

        Args:
            chunk_size: Optional override for splitter chunk size.
            chunk_overlap: Optional override for splitter chunk overlap size.
            embedding_model: Optional override for embedding model name.
            llm_model: Optional override for LLM model name.
            target_max_input_tokens: Optional override for input budget before model clamp.
            target_max_output_tokens: Optional override for output budget before model clamp.
            target_max_context_tokens: Optional override for retrieval-context token budget.
            input_budget_utilization_ratio: Optional override for input-capacity utilization ratio.
            context_budget_utilization_ratio: Optional override for context budget utilization ratio.
            full_text_input_budget_utilization_ratio: Optional override for global full-text input ratio.
            full_text_context_budget_utilization_ratio: Optional override for global full-text context ratio.
            embedding_batch_size: Optional override for index-time embedding batch size.
            bundle_cache_capacity: Optional override for in-memory bundle cache capacity.
            session_recent_limit: Optional override for per-session recent history length.
            base_near_chunk_threshold: Optional override for local-reading base threshold.
            min_near_chunk_threshold: Optional override for local-reading threshold lower bound.
            max_near_chunk_threshold: Optional override for local-reading threshold upper bound.
            global_scope_min_top_k: Optional override for global-scope minimum retrieval top_k.
            global_coverage_chunk_gap: Optional override for global coverage dedup chunk gap.
        """
        override_values: dict[str, int | float | str] = {}
        if chunk_size is not None:
            override_values["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            override_values["chunk_overlap"] = chunk_overlap
        if embedding_model is not None:
            override_values["embedding_model"] = embedding_model
        if llm_model is not None:
            override_values["llm_model"] = llm_model
        if target_max_input_tokens is not None:
            override_values["target_max_input_tokens"] = target_max_input_tokens
        if target_max_output_tokens is not None:
            override_values["target_max_output_tokens"] = target_max_output_tokens
        if target_max_context_tokens is not None:
            override_values["target_max_context_tokens"] = target_max_context_tokens
        if input_budget_utilization_ratio is not None:
            override_values["input_budget_utilization_ratio"] = input_budget_utilization_ratio
        if context_budget_utilization_ratio is not None:
            override_values["context_budget_utilization_ratio"] = context_budget_utilization_ratio
        if full_text_input_budget_utilization_ratio is not None:
            override_values["full_text_input_budget_utilization_ratio"] = (
                full_text_input_budget_utilization_ratio
            )
        if full_text_context_budget_utilization_ratio is not None:
            override_values["full_text_context_budget_utilization_ratio"] = (
                full_text_context_budget_utilization_ratio
            )
        if embedding_batch_size is not None:
            override_values["embedding_batch_size"] = embedding_batch_size
        if bundle_cache_capacity is not None:
            override_values["bundle_cache_capacity"] = bundle_cache_capacity
        if session_recent_limit is not None:
            override_values["session_recent_limit"] = session_recent_limit
        if base_near_chunk_threshold is not None:
            override_values["base_near_chunk_threshold"] = base_near_chunk_threshold
        if min_near_chunk_threshold is not None:
            override_values["min_near_chunk_threshold"] = min_near_chunk_threshold
        if max_near_chunk_threshold is not None:
            override_values["max_near_chunk_threshold"] = max_near_chunk_threshold
        if global_scope_min_top_k is not None:
            override_values["global_scope_min_top_k"] = global_scope_min_top_k
        if global_coverage_chunk_gap is not None:
            override_values["global_coverage_chunk_gap"] = global_coverage_chunk_gap

        self.app_config = replace(AppDIConfig(), **override_values)

        self.container = ApplicationLookupContainer.build(self.app_config)
        self.bundle_provider = self.container.bundle_provider()
        self.session_manager = self.container.session_manager()
        self.context_orchestrator = self.container.context_orchestrator()
        self.prompt_assembler = self.container.prompt_assembler()
        self.llm_provider = self.container.llm_provider()

    def get_bundle(self, doc_name: str) -> FaissIndexBundle:
        """Ensure index/profile readiness and return query-ready bundle.

        Args:
            doc_name: Logical document name and artifact namespace.

        Returns:
            Ready ``FaissIndexBundle`` for retrieval and answering.
        """
        return self.bundle_provider.get_bundle(doc_name)

    def get_or_create_session(self, session_id: str, doc_name: str) -> ReadingSession:
        """Get existing session by id or create/reset it."""
        return self.session_manager.get_or_create_session(session_id, doc_name)

    def get_session(self, session_id: str) -> ReadingSession | None:
        """Return current session snapshot for inspection."""
        return self.session_manager.get_session(session_id)

    def ask(
        self,
        question: str,
        doc_name: str,
        top_k: int = 3,
        session_id: str | None = None,
    ) -> str:
        """Execute one QA turn and optionally advance session reading state.

        Args:
            question: Original user question text.
            doc_name: Target document name.
            top_k: Maximum number of retrieval hits to use.
            session_id: Optional session id for reading-context continuity.

        Returns:
            Answer text generated for this turn.
        """
        session: ReadingSession | None = None
        session_active_chunk_index: int | None = None
        if session_id is not None:
            session = self.session_manager.get_or_create_session(session_id, doc_name)
            session_active_chunk_index = session.active_chunk_index
            print(
                f"Coordinator#ask before: session_id={session_id}, "
                f"active_chunk_index={session_active_chunk_index}"
            )

        bundle = self.bundle_provider.get_bundle(doc_name)
        context_result = self.context_orchestrator.build(
            query=question,
            bundle=bundle,
            top_k=top_k,
            session_active_chunk_index=session_active_chunk_index,
        )
        if context_result.answer_mode.level == AnswerLevel.REJECT:
            answer_text = "Not found"
        else:
            prompt = self.prompt_assembler.build_answer_prompt(
                context=context_result.context_text,
                question=context_result.standardized_question,
                profile=bundle.profile,
                answer_mode=context_result.answer_mode,
                prompt_mode=context_result.prompt_mode,
            )
            answer_text = self.llm_provider.complete_text(prompt)

        if session is not None:
            self.session_manager.update_session(
                session=session,
                result=SessionUpdateResult(
                    question=question,
                    bundle=bundle,
                    results=context_result.results,
                ),
            )
            print(
                f"Coordinator#ask after: session_id={session.session_id}, "
                f"active_chunk_index_before={session_active_chunk_index}, "
                f"active_chunk_index_after={session.active_chunk_index}, "
                f"context_mode={context_result.prompt_mode}"
            )

        return answer_text
