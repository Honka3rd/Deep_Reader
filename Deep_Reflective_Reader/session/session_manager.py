from dataclasses import dataclass, replace
from threading import Lock
from typing import Dict

from retrieval.faiss_index_bundle import FaissIndexBundle
from session.reading_session import ReadingSession
from retrieval.search_metadata import SearchMetadata


@dataclass(frozen=True)
class SessionUpdateResult:
    """Input payload for one session update after an ask turn."""
    question: str
    bundle: FaissIndexBundle
    results: list[SearchMetadata]


class SessionManager:
    """Own in-memory session lifecycle and reading-state updates."""
    session_store: Dict[str, ReadingSession]
    session_recent_limit: int

    def __init__(
        self,
        session_recent_limit: int = 10,
    ):
        """Initialize in-memory session storage."""
        self.session_store = {}
        self.session_recent_limit = session_recent_limit
        self._lock = Lock()

    @staticmethod
    def _snapshot(session: ReadingSession) -> ReadingSession:
        # List mutations on a caller's snapshot must not mutate manager state.
        return replace(
            session,
            recent_chunk_indices=list(session.recent_chunk_indices),
            recent_questions=list(session.recent_questions),
        )

    def get_or_create_session(self, session_id: str, doc_name: str) -> ReadingSession:
        """Capture a detached snapshot; a document switch starts a new generation."""
        with self._lock:
            current = self.session_store.get(session_id)
            if current is None or current.doc_name != doc_name:
                current = ReadingSession(session_id=session_id, doc_name=doc_name)
                self.session_store[session_id] = current
            return self._snapshot(current)

    def get_session(self, session_id: str) -> ReadingSession | None:
        """Read a detached snapshot without exposing mutable manager history."""
        with self._lock:
            current = self.session_store.get(session_id)
            return self._snapshot(current) if current is not None else None

    def reset_session(self, session_id: str) -> None:
        """Clear progress and invalidate pending turns, including same-document ones."""
        with self._lock:
            current = self.session_store.get(session_id)
            if current is not None:
                self.session_store[session_id] = ReadingSession(
                    session_id=current.session_id, doc_name=current.doc_name,
                )

    @staticmethod
    def _extract_chunk_index(bundle: FaissIndexBundle, result: SearchMetadata) -> int | None:
        """Resolve chunk index for a retrieval hit."""
        record = bundle.id_to_record.get(result.faiss_id)
        if record is None:
            return None

        chunk_index = record.chunk_index()
        if isinstance(chunk_index, int):
            return chunk_index
        return None

    def update_session(
        self,
        session: ReadingSession,
        result: SessionUpdateResult,
    ) -> bool:
        """Merge a completion into its current generation, or discard a stale turn.

        No model/retrieval work occurs under the short compare-and-update lock.
        Same-generation turns merge in completion order rather than replacing
        current history with the request's older snapshot.
        """
        best_chunk_index = (
            self._extract_chunk_index(result.bundle, result.results[0])
            if result.results else None
        )
        with self._lock:
            current = self.session_store.get(session.session_id)
            if (
                current is None
                or current.doc_name != session.doc_name
                or current.generation != session.generation
            ):
                return False

            questions = (current.recent_questions + [result.question])[-self.session_recent_limit:]
            chunks = current.recent_chunk_indices
            anchor = current.active_chunk_index
            if best_chunk_index is not None:
                anchor = best_chunk_index
                chunks = (chunks + [best_chunk_index])[-self.session_recent_limit:]
            self.session_store[session.session_id] = replace(
                current, active_chunk_index=anchor,
                recent_questions=questions, recent_chunk_indices=chunks,
            )
            return True
