from dataclasses import dataclass
from typing import Any
from llm.openai_llm_provider import OpenAIModelName


@dataclass(frozen=True)
class RetrievalTokenBudgetConfig:
    """Token budget policy for retrieval and full-text orchestration."""
    target_max_input_tokens: int
    target_max_output_tokens: int
    target_max_context_tokens: int
    input_budget_utilization_ratio: float
    context_budget_utilization_ratio: float
    full_text_input_budget_utilization_ratio: float
    full_text_context_budget_utilization_ratio: float


@dataclass(frozen=True)
class PromptTextNormalizationConfig:
    """Generic prompt-text normalization policy based on model capability."""
    default_excerpt_chars: int
    min_excerpt_chars: int
    max_excerpt_chars_hard_cap: int
    chars_per_token: int


@dataclass(frozen=True)
class ProfilePromptPolicyConfig:
    """Profile-specific prompt policy (topic/summary overhead + utilization)."""
    topic_prompt_overhead_tokens: int
    summary_prompt_overhead_tokens: int
    structure_prompt_overhead_tokens: int
    topic_excerpt_token_utilization_ratio: float
    summary_excerpt_token_utilization_ratio: float
    structure_excerpt_token_utilization_ratio: float
    evidence_head_chars: int
    evidence_tail_chars: int
    evidence_middle_chars: int
    evidence_heading_lines_limit: int


@dataclass(frozen=True)
class QuestionScopePolicyConfig:
    """Scope-resolution policy values for semantic/LLM fallback routing."""
    global_similarity_threshold: float
    llm_gray_zone_min_similarity: float
    llm_gray_zone_max_similarity: float
    llm_fallback_enabled: bool
    llm_summary_char_limit: int
    local_anchor_similarity_threshold: float


@dataclass(frozen=True)
class TaskUnitSplitPolicyConfig:
    """Task-unit split and semantic boundary scoring policy values."""
    min_chars: int
    max_chars: int
    split_mode: str
    semantic_boundary_window_chars: int
    semantic_context_window_chars: int
    semantic_top_k_candidates: int
    semantic_top_k_candidates_max: int
    semantic_max_scoring_per_window: int
    semantic_max_scoring_per_section: int
    semantic_score_weight: float
    semantic_scoring_debug_log: bool
    semantic_embedding_batch_size: int
    semantic_embedding_cache_size: int


@dataclass(frozen=True)
class EnhancedParsePolicyConfig:
    """Enhanced parser recommendation policy thresholds and bounds."""
    min_section_count: int
    max_section_count: int
    min_title_coverage: float
    min_sections_for_ratio_signal: int
    min_units_for_ratio_signal: int
    max_affected_section_ratio: float
    max_fallback_task_unit_ratio: float
    min_avg_chars_per_section: int
    max_avg_chars_per_section: int
    min_raw_chars_for_structure_density_signal: int
    recommend_score_threshold: int


@dataclass(frozen=True)
class LLMSectionPreviewPolicyConfig:
    """LLM section-split prompt preview policy."""
    prompt_overhead_tokens: int
    excerpt_token_utilization_ratio: float
    preview_head_ratio: float
    preview_omitted_separator: str


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
    # Default excerpt character budget for prompt-conditioning document profile generation.
    # Introduced to provide deterministic fallback when model capability is unavailable.
    profile_prompt_default_excerpt_chars: int = 16_000
    # Minimum excerpt character budget for document profile prompts under capability scaling.
    # Introduced to avoid over-truncation on small-capability models.
    profile_prompt_min_excerpt_chars: int = 2_000
    # Hard cap for profile prompt excerpt characters even on very large-context models.
    # Introduced to prevent unbounded prompt growth and protect latency/cost.
    profile_prompt_max_excerpt_chars_hard_cap: int = 400_000
    # Heuristic chars-per-token ratio used for prompt excerpt budget conversion.
    # Introduced to keep budgeting model-agnostic while remaining conservative.
    profile_prompt_chars_per_token: int = 4
    # Estimated fixed token overhead for topic classification prompt instructions.
    # Introduced so profile excerpt budget can reserve instruction/headroom.
    profile_topic_prompt_overhead_tokens: int = 700
    # Estimated fixed token overhead for summary-generation prompt instructions.
    # Introduced so profile excerpt budget can reserve instruction/headroom.
    profile_summary_prompt_overhead_tokens: int = 900
    # Estimated fixed token overhead for structure-profile JSON prompt instructions.
    # Introduced so structure-hints prompt can reserve instruction/headroom.
    profile_structure_prompt_overhead_tokens: int = 1_100
    # Utilization ratio of available prompt tokens used by topic prompt excerpt.
    # Introduced to balance topic classification signal and instruction headroom.
    profile_topic_excerpt_token_utilization_ratio: float = 0.55
    # Utilization ratio of available prompt tokens used by summary prompt excerpt.
    # Introduced to improve summary grounding under large-context models.
    profile_summary_excerpt_token_utilization_ratio: float = 0.70
    # Utilization ratio of available prompt tokens used by structure-profile prompt excerpt.
    # Introduced to provide sufficient structural signal for hints extraction.
    profile_structure_excerpt_token_utilization_ratio: float = 0.72
    # Head excerpt char limit for structure-profile evidence package.
    # Introduced to bound profile prompt while preserving leading structure signals.
    profile_evidence_head_chars: int = 4_000
    # Tail excerpt char limit for structure-profile evidence package.
    # Introduced to preserve appendix/back-matter clues in compact evidence package.
    profile_evidence_tail_chars: int = 2_500
    # Middle excerpt char limit for structure-profile evidence package.
    # Introduced to preserve mid-document structural clues in compact evidence package.
    profile_evidence_middle_chars: int = 2_200
    # Maximum heading-like lines included in structure-profile evidence package.
    # Introduced to keep profile prompt concise while retaining structural cues.
    profile_evidence_heading_lines_limit: int = 60
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
    # Character window size for boundary-side semantic snippets in task-unit split scoring.
    # Introduced to balance local semantic continuity signal and token/embedding overhead.
    task_unit_semantic_boundary_window_chars: int = 220
    # Character window size for wider context semantic snippets around candidate boundary.
    # Introduced to estimate semantic shift across boundary more robustly than punctuation only.
    task_unit_semantic_context_window_chars: int = 700
    # Maximum top-ranked heuristic boundary candidates reranked with semantic scoring per window.
    # Introduced to avoid scoring every boundary candidate and cap semantic rerank overhead.
    task_unit_semantic_top_k_candidates: int = 3
    # Maximum allowed request-time override value for semantic top-k rerank candidates.
    # Introduced to prevent API callers from setting extreme values that could hurt latency.
    task_unit_semantic_top_k_candidates_max: int = 20
    # Maximum semantic scoring attempts per split window.
    # Introduced to guarantee bounded semantic scorer calls under dense candidate windows.
    task_unit_semantic_max_scoring_per_window: int = 3
    # Maximum semantic scoring attempts per section split run.
    # Introduced to keep task-unit split latency bounded on very long sections.
    task_unit_semantic_max_scoring_per_section: int = 12
    # Weight multiplier for embedding-based semantic boundary score added to heuristic score.
    # Introduced to tune semantic influence without overriding deterministic structural rules.
    task_unit_semantic_score_weight: float = 1.0
    # Toggle compact semantic-scoring stats logs in heuristic split resolver.
    # Introduced for observability of rerank participation and budget fallbacks.
    task_unit_semantic_scoring_debug_log: bool = True
    # Batch size for semantic scorer embedding prefill.
    # Introduced to reduce per-snippet network round trips when warming semantic cache.
    task_unit_semantic_embedding_batch_size: int = 24
    # Maximum cached snippet embeddings in boundary scorer.
    # Introduced to reduce repeated embedding calls while bounding in-memory cache size.
    task_unit_semantic_embedding_cache_size: int = 256
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
    # Estimated fixed token overhead for LLM section-split planning prompt.
    # Introduced so preview excerpt budget reserves instruction/headroom.
    llm_section_preview_prompt_overhead_tokens: int = 1_200
    # Utilization ratio of available prompt tokens used by section-split preview excerpt.
    # Introduced to balance plan quality and prompt-headroom safety.
    llm_section_preview_excerpt_token_utilization_ratio: float = 0.65
    # Ratio of preview content budget allocated to document head in head+tail preview rendering.
    # Introduced to keep early ToC/front-matter visibility while still preserving tail regions.
    llm_section_preview_head_ratio: float = 0.65
    # Explicit omission marker inserted between head/tail preview slices.
    # Introduced to make truncation visible to the model in planning prompts.
    llm_section_preview_omitted_separator: str = "\n\n[... middle omitted ...]\n\n"

    def retrieval_token_budget(self) -> RetrievalTokenBudgetConfig:
        """Return grouped retrieval/full-text token budget config."""
        return RetrievalTokenBudgetConfig(
            target_max_input_tokens=self.target_max_input_tokens,
            target_max_output_tokens=self.target_max_output_tokens,
            target_max_context_tokens=self.target_max_context_tokens,
            input_budget_utilization_ratio=self.input_budget_utilization_ratio,
            context_budget_utilization_ratio=self.context_budget_utilization_ratio,
            full_text_input_budget_utilization_ratio=(
                self.full_text_input_budget_utilization_ratio
            ),
            full_text_context_budget_utilization_ratio=(
                self.full_text_context_budget_utilization_ratio
            ),
        )

    def prompt_text_normalization(self) -> PromptTextNormalizationConfig:
        """Return generic prompt-text normalization config."""
        return PromptTextNormalizationConfig(
            default_excerpt_chars=self.profile_prompt_default_excerpt_chars,
            min_excerpt_chars=self.profile_prompt_min_excerpt_chars,
            max_excerpt_chars_hard_cap=self.profile_prompt_max_excerpt_chars_hard_cap,
            chars_per_token=self.profile_prompt_chars_per_token,
        )

    def profile_prompt_policy(self) -> ProfilePromptPolicyConfig:
        """Return profile-specific prompt policy config."""
        return ProfilePromptPolicyConfig(
            topic_prompt_overhead_tokens=self.profile_topic_prompt_overhead_tokens,
            summary_prompt_overhead_tokens=self.profile_summary_prompt_overhead_tokens,
            structure_prompt_overhead_tokens=self.profile_structure_prompt_overhead_tokens,
            topic_excerpt_token_utilization_ratio=(
                self.profile_topic_excerpt_token_utilization_ratio
            ),
            summary_excerpt_token_utilization_ratio=(
                self.profile_summary_excerpt_token_utilization_ratio
            ),
            structure_excerpt_token_utilization_ratio=(
                self.profile_structure_excerpt_token_utilization_ratio
            ),
            evidence_head_chars=self.profile_evidence_head_chars,
            evidence_tail_chars=self.profile_evidence_tail_chars,
            evidence_middle_chars=self.profile_evidence_middle_chars,
            evidence_heading_lines_limit=self.profile_evidence_heading_lines_limit,
        )

    def question_scope_policy(self) -> QuestionScopePolicyConfig:
        """Return grouped question-scope policy config."""
        return QuestionScopePolicyConfig(
            global_similarity_threshold=self.question_scope_global_similarity_threshold,
            llm_gray_zone_min_similarity=self.question_scope_llm_gray_zone_min_similarity,
            llm_gray_zone_max_similarity=self.question_scope_llm_gray_zone_max_similarity,
            llm_fallback_enabled=self.question_scope_llm_fallback_enabled,
            llm_summary_char_limit=self.question_scope_llm_summary_char_limit,
            local_anchor_similarity_threshold=(
                self.question_scope_local_anchor_similarity_threshold
            ),
        )

    def task_unit_split_policy(self) -> TaskUnitSplitPolicyConfig:
        """Return grouped task-unit split policy config."""
        return TaskUnitSplitPolicyConfig(
            min_chars=self.task_unit_min_chars,
            max_chars=self.task_unit_max_chars,
            split_mode=self.task_unit_split_mode,
            semantic_boundary_window_chars=self.task_unit_semantic_boundary_window_chars,
            semantic_context_window_chars=self.task_unit_semantic_context_window_chars,
            semantic_top_k_candidates=self.task_unit_semantic_top_k_candidates,
            semantic_top_k_candidates_max=self.task_unit_semantic_top_k_candidates_max,
            semantic_max_scoring_per_window=self.task_unit_semantic_max_scoring_per_window,
            semantic_max_scoring_per_section=self.task_unit_semantic_max_scoring_per_section,
            semantic_score_weight=self.task_unit_semantic_score_weight,
            semantic_scoring_debug_log=self.task_unit_semantic_scoring_debug_log,
            semantic_embedding_batch_size=self.task_unit_semantic_embedding_batch_size,
            semantic_embedding_cache_size=self.task_unit_semantic_embedding_cache_size,
        )

    def enhanced_parse_policy(self) -> EnhancedParsePolicyConfig:
        """Return grouped enhanced-parse recommendation policy config."""
        return EnhancedParsePolicyConfig(
            min_section_count=self.enhanced_parse_min_section_count,
            max_section_count=self.enhanced_parse_max_section_count,
            min_title_coverage=self.enhanced_parse_min_title_coverage,
            min_sections_for_ratio_signal=self.enhanced_parse_min_sections_for_ratio_signal,
            min_units_for_ratio_signal=self.enhanced_parse_min_units_for_ratio_signal,
            max_affected_section_ratio=self.enhanced_parse_max_affected_section_ratio,
            max_fallback_task_unit_ratio=self.enhanced_parse_max_fallback_task_unit_ratio,
            min_avg_chars_per_section=self.enhanced_parse_min_avg_chars_per_section,
            max_avg_chars_per_section=self.enhanced_parse_max_avg_chars_per_section,
            min_raw_chars_for_structure_density_signal=(
                self.enhanced_parse_min_raw_chars_for_structure_density_signal
            ),
            recommend_score_threshold=self.enhanced_parse_recommend_score_threshold,
        )

    def llm_section_preview_policy(self) -> LLMSectionPreviewPolicyConfig:
        """Return grouped LLM section-split preview policy config."""
        return LLMSectionPreviewPolicyConfig(
            prompt_overhead_tokens=self.llm_section_preview_prompt_overhead_tokens,
            excerpt_token_utilization_ratio=(
                self.llm_section_preview_excerpt_token_utilization_ratio
            ),
            preview_head_ratio=self.llm_section_preview_head_ratio,
            preview_omitted_separator=self.llm_section_preview_omitted_separator,
        )

    def to_container_config_dict(self) -> dict[str, Any]:
        """Export normalized container config map from grouped policy views."""
        retrieval_budget = self.retrieval_token_budget()
        prompt_text_normalization = self.prompt_text_normalization()
        profile_prompt_policy = self.profile_prompt_policy()
        scope_policy = self.question_scope_policy()
        task_unit_policy = self.task_unit_split_policy()
        enhanced_parse_policy = self.enhanced_parse_policy()
        llm_section_preview_policy = self.llm_section_preview_policy()
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "target_max_input_tokens": retrieval_budget.target_max_input_tokens,
            "target_max_output_tokens": retrieval_budget.target_max_output_tokens,
            "target_max_context_tokens": retrieval_budget.target_max_context_tokens,
            "input_budget_utilization_ratio": retrieval_budget.input_budget_utilization_ratio,
            "context_budget_utilization_ratio": retrieval_budget.context_budget_utilization_ratio,
            "full_text_input_budget_utilization_ratio": (
                retrieval_budget.full_text_input_budget_utilization_ratio
            ),
            "full_text_context_budget_utilization_ratio": (
                retrieval_budget.full_text_context_budget_utilization_ratio
            ),
            "profile_prompt_default_excerpt_chars": (
                prompt_text_normalization.default_excerpt_chars
            ),
            "profile_prompt_min_excerpt_chars": prompt_text_normalization.min_excerpt_chars,
            "profile_prompt_max_excerpt_chars_hard_cap": (
                prompt_text_normalization.max_excerpt_chars_hard_cap
            ),
            "profile_prompt_chars_per_token": prompt_text_normalization.chars_per_token,
            "profile_topic_prompt_overhead_tokens": (
                profile_prompt_policy.topic_prompt_overhead_tokens
            ),
            "profile_summary_prompt_overhead_tokens": (
                profile_prompt_policy.summary_prompt_overhead_tokens
            ),
            "profile_structure_prompt_overhead_tokens": (
                profile_prompt_policy.structure_prompt_overhead_tokens
            ),
            "profile_topic_excerpt_token_utilization_ratio": (
                profile_prompt_policy.topic_excerpt_token_utilization_ratio
            ),
            "profile_summary_excerpt_token_utilization_ratio": (
                profile_prompt_policy.summary_excerpt_token_utilization_ratio
            ),
            "profile_structure_excerpt_token_utilization_ratio": (
                profile_prompt_policy.structure_excerpt_token_utilization_ratio
            ),
            "profile_evidence_head_chars": profile_prompt_policy.evidence_head_chars,
            "profile_evidence_tail_chars": profile_prompt_policy.evidence_tail_chars,
            "profile_evidence_middle_chars": profile_prompt_policy.evidence_middle_chars,
            "profile_evidence_heading_lines_limit": (
                profile_prompt_policy.evidence_heading_lines_limit
            ),
            "prompt_text_normalization": prompt_text_normalization,
            "profile_prompt_policy": profile_prompt_policy,
            "embedding_batch_size": self.embedding_batch_size,
            "bundle_cache_capacity": self.bundle_cache_capacity,
            "session_recent_limit": self.session_recent_limit,
            "base_near_chunk_threshold": self.base_near_chunk_threshold,
            "min_near_chunk_threshold": self.min_near_chunk_threshold,
            "max_near_chunk_threshold": self.max_near_chunk_threshold,
            "global_scope_min_top_k": self.global_scope_min_top_k,
            "global_coverage_chunk_gap": self.global_coverage_chunk_gap,
            "question_scope_global_similarity_threshold": (
                scope_policy.global_similarity_threshold
            ),
            "question_scope_llm_gray_zone_min_similarity": (
                scope_policy.llm_gray_zone_min_similarity
            ),
            "question_scope_llm_gray_zone_max_similarity": (
                scope_policy.llm_gray_zone_max_similarity
            ),
            "question_scope_llm_fallback_enabled": scope_policy.llm_fallback_enabled,
            "question_scope_llm_summary_char_limit": scope_policy.llm_summary_char_limit,
            "question_scope_local_anchor_similarity_threshold": (
                scope_policy.local_anchor_similarity_threshold
            ),
            "quiz_min_section_chars": self.quiz_min_section_chars,
            "section_task_topic_semantic_match_enabled": (
                self.section_task_topic_semantic_match_enabled
            ),
            "section_task_topic_semantic_similarity_threshold": (
                self.section_task_topic_semantic_similarity_threshold
            ),
            "task_unit_min_chars": task_unit_policy.min_chars,
            "task_unit_max_chars": task_unit_policy.max_chars,
            "task_unit_split_mode": task_unit_policy.split_mode,
            "task_unit_semantic_boundary_window_chars": (
                task_unit_policy.semantic_boundary_window_chars
            ),
            "task_unit_semantic_context_window_chars": (
                task_unit_policy.semantic_context_window_chars
            ),
            "task_unit_semantic_top_k_candidates": (
                task_unit_policy.semantic_top_k_candidates
            ),
            "task_unit_semantic_top_k_candidates_max": (
                task_unit_policy.semantic_top_k_candidates_max
            ),
            "task_unit_semantic_max_scoring_per_window": (
                task_unit_policy.semantic_max_scoring_per_window
            ),
            "task_unit_semantic_max_scoring_per_section": (
                task_unit_policy.semantic_max_scoring_per_section
            ),
            "task_unit_semantic_score_weight": task_unit_policy.semantic_score_weight,
            "task_unit_semantic_scoring_debug_log": (
                task_unit_policy.semantic_scoring_debug_log
            ),
            "task_unit_semantic_embedding_batch_size": (
                task_unit_policy.semantic_embedding_batch_size
            ),
            "task_unit_semantic_embedding_cache_size": (
                task_unit_policy.semantic_embedding_cache_size
            ),
            "enhanced_parse_min_section_count": enhanced_parse_policy.min_section_count,
            "enhanced_parse_max_section_count": enhanced_parse_policy.max_section_count,
            "enhanced_parse_min_title_coverage": enhanced_parse_policy.min_title_coverage,
            "enhanced_parse_min_sections_for_ratio_signal": (
                enhanced_parse_policy.min_sections_for_ratio_signal
            ),
            "enhanced_parse_min_units_for_ratio_signal": (
                enhanced_parse_policy.min_units_for_ratio_signal
            ),
            "enhanced_parse_max_affected_section_ratio": (
                enhanced_parse_policy.max_affected_section_ratio
            ),
            "enhanced_parse_max_fallback_task_unit_ratio": (
                enhanced_parse_policy.max_fallback_task_unit_ratio
            ),
            "enhanced_parse_min_avg_chars_per_section": (
                enhanced_parse_policy.min_avg_chars_per_section
            ),
            "enhanced_parse_max_avg_chars_per_section": (
                enhanced_parse_policy.max_avg_chars_per_section
            ),
            "enhanced_parse_min_raw_chars_for_structure_density_signal": (
                enhanced_parse_policy.min_raw_chars_for_structure_density_signal
            ),
            "enhanced_parse_recommend_score_threshold": (
                enhanced_parse_policy.recommend_score_threshold
            ),
            "llm_section_preview_policy": llm_section_preview_policy,
        }
