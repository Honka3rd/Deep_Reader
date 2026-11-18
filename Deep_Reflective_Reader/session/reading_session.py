from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class ReadingSession:
    """Detached reading snapshot with immutable lifecycle identity."""
    session_id: str
    doc_name: str
    active_chunk_index: int | None = None
    recent_chunk_indices: list[int] = field(default_factory=list)
    recent_questions: list[str] = field(default_factory=list)
    generation: str = field(default_factory=lambda: uuid4().hex)
