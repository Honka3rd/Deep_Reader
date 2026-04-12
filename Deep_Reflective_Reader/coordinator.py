from app_DI_config import AppDIConfig
from container import ApplicationLookupContainer
from faiss_index_bundle import FaissIndexBundle
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
        """
        self.app_config = AppDIConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            llm_model=llm_model,
            embedding_batch_size=embedding_batch_size,
            bundle_cache_capacity=bundle_cache_capacity,
            session_recent_limit=session_recent_limit,
        )

        self.container = ApplicationLookupContainer.build(self.app_config)
        self.bundle_provider = self.container.bundle_provider()
        self.session_manager = self.container.session_manager()

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
        answer_text, results, prompt_mode = bundle.answer_with_trace(
            query=question,
            top_k=top_k,
            session_active_chunk_index=session_active_chunk_index,
            near_chunk_threshold=2,
            local_window_radius=1,
        )

        if session is not None:
            self.session_manager.update_session(
                session=session,
                result=SessionUpdateResult(
                    question=question,
                    bundle=bundle,
                    results=results,
                ),
            )
            print(
                f"Coordinator#ask after: session_id={session.session_id}, "
                f"active_chunk_index_before={session_active_chunk_index}, "
                f"active_chunk_index_after={session.active_chunk_index}, "
                f"context_mode={prompt_mode}"
            )

        return answer_text
