from dataclasses import dataclass

@dataclass(frozen=True)
class AppDIConfig:
    chunk_size: int = 300
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    embedding_batch_size: int = 64
    bundle_cache_capacity: int = 3