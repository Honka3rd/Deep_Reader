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
    # Similarity threshold for semantic keyword -> global scope (without LLM fallback).
    # Introduced to keep deterministic global classification when semantic confidence is strong.
    question_scope_global_similarity_threshold: float = 0.78
    # Lower bound of semantic gray zone that can trigger LLM fallback scope classification.
    # Introduced so moderately ambiguous questions can still be classified as global.
    question_scope_llm_gray_zone_min_similarity: float = 0.30
    # Upper bound of semantic gray zone for LLM fallback (exclusive upper bound).
    # Introduced to avoid LLM calls when semantic confidence is already clearly global.
    question_scope_llm_gray_zone_max_similarity: float = 0.78
    # Toggle to enable/disable LLM fallback scope classification in semantic gray zone.
    # Introduced to allow cost/latency control without code changes.
    question_scope_llm_fallback_enabled: bool = True
    # Max characters from document summary passed to LLM fallback scope classifier.
    # Introduced to bound token cost while preserving enough global context signal.
    question_scope_llm_summary_char_limit: int = 800
    # Similarity threshold for session-local anchor semantic matching in scope resolver.
    # Introduced to improve local-reference detection beyond lexical string matching.
    question_scope_local_anchor_similarity_threshold: float = 0.75
    # Minimum section/chapter content length (characters) required before quiz generation.
    # Introduced to skip low-signal quiz requests on very short content.
    quiz_min_section_chars: int = 400
    # Toggle for semantic topic-guidance matching in section-task prompts.
    # Introduced so topic guidance can use embeddings when lexical match is weak.
    section_task_topic_semantic_match_enabled: bool = True
    # Similarity threshold for semantic topic-guidance rule selection.
    # Introduced to keep topic guidance conservative and avoid noisy semantic matches.
    section_task_topic_semantic_similarity_threshold: float = 0.78
    # Minimum content length per task unit before considering adjacent-merge fallback.
    # Introduced to avoid tiny fragmented units in section-task execution.
    task_unit_min_chars: int = 300
    # Maximum preferred content length per task unit for fallback splitting.
    # Introduced to prevent single giant units from degrading task quality.
    task_unit_max_chars: int = 1600
    # Split strategy for oversized single-section internal task-unit segmentation.
    # Introduced to switch between low-cost semantic-safe and progressive split policies.
    task_unit_split_mode: str = "semantic_safe"
    # Minimum reasonable section count; below this structure is likely too coarse.
    # Introduced as a strong signal for recommending enhanced parser.
    enhanced_parse_min_section_count: int = 3
    # Maximum reasonable section count; above this structure is likely too fragmented.
    # Introduced as a strong signal for recommending enhanced parser.
    enhanced_parse_max_section_count: int = 220
    # Minimum expected title coverage ratio in common parse output.
    # Introduced to detect low-structure readability (too many untitled sections).
    enhanced_parse_min_title_coverage: float = 0.55
    # Minimum sections required before ratio-type weak signals are considered.
    # Introduced to avoid noisy recommendation on very small documents.
    enhanced_parse_min_sections_for_ratio_signal: int = 8
    # Minimum task units required before fallback-unit ratio weak signal is considered.
    # Introduced to avoid unstable trigger decisions on tiny layouts.
    enhanced_parse_min_units_for_ratio_signal: int = 8
    # Upper bound for affected section ratio (split/merged) before weak signal is added.
    # Introduced to capture excessive task-time fallback intervention.
    enhanced_parse_max_affected_section_ratio: float = 0.55
    # Upper bound for fallback-generated task unit ratio before weak signal is added.
    # Introduced to capture heavy resolver fallback usage.
    enhanced_parse_max_fallback_task_unit_ratio: float = 0.55
    # Minimum average chars per section for long-enough documents.
    # Introduced to detect over-fragmented section structures with length normalization.
    enhanced_parse_min_avg_chars_per_section: int = 180
    # Maximum average chars per section for long-enough documents.
    # Introduced to detect over-coarse section structures with length normalization.
    enhanced_parse_max_avg_chars_per_section: int = 8000
    # Minimum raw-text length before avg-chars-per-section structure-density signal applies.
    # Introduced to avoid noisy abnormality triggers on tiny documents.
    enhanced_parse_min_raw_chars_for_structure_density_signal: int = 2000
    # Score threshold for recommending enhanced parser.
    # Introduced to combine strong/weak signals into one deterministic decision.
    enhanced_parse_recommend_score_threshold: int = 4
