from dataclasses import dataclass
from llm.openai_llm_provider import OpenAIModelName

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
    llm_model: OpenAIModelName | str = OpenAIModelName.GPT_4_1_MINI
    # App-level target for max prompt/input tokens in normal retrieval/local paths.
    # Introduced as fallback when model capability is unavailable or when retrieval budget should stay conservative.
    target_max_input_tokens: int = 3200
    # App-level target for max generated tokens in normal retrieval/local paths.
    # Introduced to control output length/cost with a conservative default policy.
    target_max_output_tokens: int = 500
    # App-level target for retrieval/local context token budget (subset of input budget).
    # Introduced to tune evidence size independently from total prompt budget.
    target_max_context_tokens: int = 1500
    # Conservative utilization ratio for retrieval/local input budget when capability is known.
    # Introduced to avoid retrieval path over-consuming model context.
    input_budget_utilization_ratio: float = 0.2
    # Conservative utilization ratio from retrieval/local input budget to retrieval/local context budget.
    # Introduced to preserve non-context prompt headroom in normal modes.
    context_budget_utilization_ratio: float = 0.9
    # Global full-text input-budget utilization ratio against model max input tokens.
    # Introduced to let full-text mode use near model capability with safety headroom.
    full_text_input_budget_utilization_ratio: float = 0.5
    # Global full-text context-budget utilization ratio derived from full-text input budget.
    # Introduced to cap full-text context conservatively while remaining much broader than retrieval.
    full_text_context_budget_utilization_ratio: float = 0.7
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
