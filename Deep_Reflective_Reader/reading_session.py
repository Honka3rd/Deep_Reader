from dataclasses import dataclass, field


@dataclass
class ReadingSession:
    session_id: str
    doc_name: str
    active_chunk_index: int | None = None
    recent_chunk_indices: list[int] = field(default_factory=list)
    recent_questions: list[str] = field(default_factory=list)
