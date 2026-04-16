from dataclasses import dataclass

@dataclass(frozen=True)
class AppDIConfig:
    """Application-level DI configuration values."""
    # Max characters/tokens per chunk before indexing.
    # Introduced to balance retrieval granularity and semantic completeness.
    chunk_size: int = 300
    # Overlap size between adjacent chunks.
    # Introduced to reduce boundary information loss across chunk splits.
    chunk_overlap: int = 50
    # Embedding model used for vector indexing and similarity search.
    # Introduced to centralize embedding behavior and make model swap configurable.
    embedding_model: str = "text-embedding-3-small"
    # Chat/completion model used for answer generation.
    # Introduced so answer quality/cost/latency tradeoff can be tuned without code changes.
    llm_model: str = "gpt-4.1-mini"
    # Batch size for embedding calls during index build.
    # Introduced to control build throughput and API pressure.
    embedding_batch_size: int = 64
    # In-memory cache size for runtime bundles.
    # Introduced to avoid repeated index/profile assembly while bounding memory usage.
    bundle_cache_capacity: int = 3
    # Max number of recent questions/chunk indices kept per session.
    # Introduced to preserve short reading history while preventing unbounded session growth.
    session_recent_limit: int = 10
    # Default proximity threshold for local-reading continuity checks.
    # Introduced as the baseline for dynamic near-chunk decision.
    base_near_chunk_threshold: int = 2
    # Lower bound for dynamic near-chunk threshold.
    # Introduced to prevent overly strict or negative threshold behavior.
    min_near_chunk_threshold: int = 1
    # Upper bound for dynamic near-chunk threshold.
    # Introduced to prevent local-reading expansion from becoming too permissive.
    max_near_chunk_threshold: int = 4
    # Minimum retrieval top_k when question scope is global.
    # Introduced to improve evidence breadth for full-document/listing questions.
    global_scope_min_top_k: int = 8
    # Neighbor chunk distance for global coverage dedup (coverage-oriented builder).
    # Introduced to reduce same-region evidence clustering and improve document-wide coverage.
    global_coverage_chunk_gap: int = 2
