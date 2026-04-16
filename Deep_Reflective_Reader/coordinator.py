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
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            embedding_model: str = "text-embedding-3-small",
            llm_model: str = "gpt-4.1-mini",
            embedding_batch_size: int = 64,
            bundle_cache_capacity: int = 3,
            session_recent_limit: int = 10,
            base_near_chunk_threshold: int = 2,
            min_near_chunk_threshold: int = 1,
            max_near_chunk_threshold: int = 4,
            global_scope_min_top_k: int = 8,
            global_coverage_chunk_gap: int = 2,
    ):
        """Initialize runtime dependencies and in-memory session storage.

        Args:
            chunk_size: Splitter chunk size.
            chunk_overlap: Splitter chunk overlap size.
            embedding_model: Embedding model name.
            llm_model: LLM model name.
            embedding_batch_size: Batch size for index-time embedding calls.
            bundle_cache_capacity: Maximum bundle objects kept in memory cache.
            session_recent_limit: Max items kept in per-session recent history.
            base_near_chunk_threshold: Base threshold for local-reading gate.
            min_near_chunk_threshold: Lower bound for dynamic local-reading threshold.
            max_near_chunk_threshold: Upper bound for dynamic local-reading threshold.
            global_scope_min_top_k: Minimum retrieval top_k when scope is global.
            global_coverage_chunk_gap: Chunk-index neighborhood distance for global coverage dedup.
        """
        self.app_config = AppDIConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            llm_model=llm_model,
            embedding_batch_size=embedding_batch_size,
            bundle_cache_capacity=bundle_cache_capacity,
            session_recent_limit=session_recent_limit,
            base_near_chunk_threshold=base_near_chunk_threshold,
            min_near_chunk_threshold=min_near_chunk_threshold,
            max_near_chunk_threshold=max_near_chunk_threshold,
            global_scope_min_top_k=global_scope_min_top_k,
            global_coverage_chunk_gap=global_coverage_chunk_gap,
        )

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
