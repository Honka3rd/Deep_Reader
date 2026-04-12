from dataclasses import dataclass

@dataclass(frozen=True)
class AppDIConfig:
    """Application-level DI configuration values."""
    chunk_size: int = 300
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    embedding_batch_size: int = 64
    bundle_cache_capacity: int = 3
    session_recent_limit: int = 10
    base_near_chunk_threshold: int = 2
    min_near_chunk_threshold: int = 1
    max_near_chunk_threshold: int = 4
