from dataclasses import dataclass
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

    def get_or_create_session(self, session_id: str, doc_name: str) -> ReadingSession:
        """Get existing session by id or create/reset it."""
        session = self.session_store.get(session_id)
        if session is None:
            session = ReadingSession(
                session_id=session_id,
                doc_name=doc_name,
            )
            self.session_store[session_id] = session
            print(
                f"Coordinator#session create: session_id={session_id}, "
                f"doc_name={doc_name}"
            )
            return session

        if session.doc_name != doc_name:
            # Reuse session id, but reset reading state for the new document.
            old_doc_name = session.doc_name
            session.doc_name = doc_name
            session.active_chunk_index = None
            session.recent_chunk_indices.clear()
            session.recent_questions.clear()
            print(
                f"Coordinator#session reset: session_id={session_id}, "
                f"old_doc_name={old_doc_name}, new_doc_name={doc_name}"
            )
        else:
            print(
                f"Coordinator#session hit: session_id={session_id}, doc_name={doc_name}"
            )

        return session

    def get_session(self, session_id: str) -> ReadingSession | None:
        """Return current session snapshot for inspection."""
        return self.session_store.get(session_id)

    def reset_session(self, session_id: str) -> None:
        """Reset reading progress for an existing session id."""
        session = self.session_store.get(session_id)
        if session is None:
            return

        old_doc_name = session.doc_name
        session.active_chunk_index = None
        session.recent_chunk_indices.clear()
        session.recent_questions.clear()
        print(
            f"Coordinator#session reset: session_id={session_id}, "
            f"old_doc_name={old_doc_name}, new_doc_name={session.doc_name}"
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
    ) -> None:
        """Update session reading state after one successful ask call."""
        session.recent_questions.append(result.question)
        if len(session.recent_questions) > self.session_recent_limit:
            session.recent_questions = session.recent_questions[-self.session_recent_limit:]

        if not result.results:
            return

        best_result = result.results[0]
        best_chunk_index = self._extract_chunk_index(result.bundle, best_result)
        if best_chunk_index is None:
            return

        session.active_chunk_index = best_chunk_index
        session.recent_chunk_indices.append(best_chunk_index)
        if len(session.recent_chunk_indices) > self.session_recent_limit:
            session.recent_chunk_indices = session.recent_chunk_indices[-self.session_recent_limit:]
