import math
from dataclasses import dataclass
from typing import Any

from coverage_oriented_context_builder import CoverageOrientedContextBuilder
from evaluated_answer.answer_mode import AnswerMode
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator
from faiss_index_bundle import FaissIndexBundle
from question_scope_resolver import QuestionScopeResolver, QuestionScopeResolution
from qa_enums import AnswerLevel, ContextMode, PromptMode, QuestionScope
from search_metadata import SearchMetadata
from standardized.question_standardizer import QuestionStandardizer
from standardized.standardized_question import StandardizedQuestion


@dataclass(frozen=True)
class ContextBuildResult:
    """Structured context-selection output for one ask turn."""
    context_text: str
    mode: ContextMode
    prompt_mode: PromptMode
    metadata: dict[str, Any]
    standardized_question: StandardizedQuestion
    answer_mode: AnswerMode
    results: list[SearchMetadata]


class ContextOrchestrator:
    """Decide context strategy (full/local/retrieval) for a QA turn."""
    question_standardizer: QuestionStandardizer
    relevance_evaluator: QuestionRelevanceEvaluator
    question_scope_resolver: QuestionScopeResolver
    global_coverage_context_builder: CoverageOrientedContextBuilder
    base_near_chunk_threshold: int
    min_near_chunk_threshold: int
    max_near_chunk_threshold: int
    global_scope_min_top_k: int
    full_text_input_budget_utilization_ratio: float
    full_text_context_budget_utilization_ratio: float

    def __init__(
        self,
        question_standardizer: QuestionStandardizer,
        relevance_evaluator: QuestionRelevanceEvaluator,
        question_scope_resolver: QuestionScopeResolver,
        global_coverage_context_builder: CoverageOrientedContextBuilder,
        base_near_chunk_threshold: int = 2,
        min_near_chunk_threshold: int = 1,
        max_near_chunk_threshold: int = 4,
        global_scope_min_top_k: int = 8,
        full_text_input_budget_utilization_ratio: float = 0.5,
        full_text_context_budget_utilization_ratio: float = 0.7,
    ):
        """Initialize orchestrator from injected strategy dependencies."""
        self.question_standardizer = question_standardizer
        self.relevance_evaluator = relevance_evaluator
        self.question_scope_resolver = question_scope_resolver
        self.global_coverage_context_builder = global_coverage_context_builder
        self.base_near_chunk_threshold = base_near_chunk_threshold
        self.min_near_chunk_threshold = min_near_chunk_threshold
        self.max_near_chunk_threshold = max_near_chunk_threshold
        self.global_scope_min_top_k = global_scope_min_top_k
        if not (0 < full_text_input_budget_utilization_ratio <= 1):
            raise ValueError("full_text_input_budget_utilization_ratio must be in (0, 1]")
        if not (0 < full_text_context_budget_utilization_ratio <= 1):
            raise ValueError("full_text_context_budget_utilization_ratio must be in (0, 1]")
        self.full_text_input_budget_utilization_ratio = full_text_input_budget_utilization_ratio
        self.full_text_context_budget_utilization_ratio = full_text_context_budget_utilization_ratio

    def _resolve_near_chunk_threshold(
        self,
        answer_mode: AnswerMode,
        results: list[SearchMetadata],
    ) -> int:
        """Dynamically resolve local-window trigger threshold per turn."""
        threshold = self.base_near_chunk_threshold
        if not results:
            return threshold

        best_score = min(result.score for result in results)
        if answer_mode.level == AnswerLevel.STRICT:
            # Stronger retrieval match => allow slightly wider local continuity.
            if best_score < 1.00:
                threshold += 1
            elif best_score > 1.08:
                threshold -= 1
        elif answer_mode.level == AnswerLevel.CAUTIOUS:
            # Cautious mode narrows local continuity gate.
            threshold -= 1

        return max(
            self.min_near_chunk_threshold,
            min(self.max_near_chunk_threshold, threshold),
        )

    @staticmethod
    def _extract_chunk_index(bundle: FaissIndexBundle, result: SearchMetadata) -> int | None:
        """Resolve chunk index for a retrieval hit."""
        record = bundle.get_node_by_id(result.faiss_id)
        if record is None:
            return None
        chunk_index = record.chunk_index()
        if isinstance(chunk_index, int):
            return chunk_index
        return None

    @staticmethod
    def _extract_texts(results: list[SearchMetadata]) -> list[str]:
        """Extract ordered chunk texts from retrieval hits."""
        return [result.text for result in results]

    def _maybe_build_full_text_context(
        self,
        bundle: FaissIndexBundle,
        question: StandardizedQuestion,
        answer_mode: AnswerMode,
        scope: QuestionScope,
    ) -> ContextBuildResult | None:
        """Try full-text mode when estimated full text fits available budget."""
        total_records = len(bundle.id_to_record)

        prompt_mode = PromptMode.FULL_TEXT
        estimated_full_text_tokens = bundle.estimate_full_text_tokens()
        effective_input_budget = bundle.max_prompt_tokens
        effective_output_budget = bundle.reserved_output_tokens
        effective_context_budget = bundle.max_context_tokens
        budget_policy = "bundle_default"
        capability_used = False

        if scope == QuestionScope.GLOBAL:
            try:
                capabilities = bundle.llm_provider.get_model_capabilities()
            except Exception as e:
                print(
                    "Warn:ContextOrchestrator#full_text_gate: "
                    f"failed to read model capabilities, fallback_to_bundle_default. error={e}"
                )
                capabilities = None

            if capabilities is not None:
                effective_input_budget = max(
                    1,
                    math.floor(
                        capabilities.max_input_tokens
                        * self.full_text_input_budget_utilization_ratio
                    ),
                )
                # Full-text path prioritizes model capability (with utilization headroom)
                # instead of retrieval-oriented app target caps.
                effective_output_budget = capabilities.max_output_tokens
                effective_context_budget = max(
                    1,
                    math.floor(
                        effective_input_budget
                        * self.full_text_context_budget_utilization_ratio
                    ),
                )
                context_budget = bundle.compute_available_context_budget_with_override(
                    question=question,
                    answer_mode=answer_mode,
                    prompt_mode=prompt_mode,
                    max_context_tokens=effective_context_budget,
                    max_prompt_tokens=effective_input_budget,
                    reserved_output_tokens=effective_output_budget,
                )
                budget_policy = "model_capability_ratio"
                capability_used = True
            else:
                context_budget = bundle.compute_available_context_budget(
                    question=question,
                    answer_mode=answer_mode,
                    prompt_mode=prompt_mode,
                )
        else:
            context_budget = bundle.compute_available_context_budget(
                question=question,
                answer_mode=answer_mode,
                prompt_mode=prompt_mode,
            )

        should_trigger_full_text = (
            total_records > 0
            and context_budget > 0
            and estimated_full_text_tokens <= context_budget
        )
        print(
            "ContextOrchestrator#full_text_gate:",
            f"scope={scope.value}",
            f"total_records={total_records}",
            f"estimated_full_text_tokens={estimated_full_text_tokens}",
            f"available_context_budget={context_budget}",
            f"effective_input_budget={effective_input_budget}",
            f"effective_output_budget={effective_output_budget}",
            f"effective_context_budget={effective_context_budget}",
            f"budget_policy={budget_policy}",
            f"capability_used={capability_used}",
            f"triggered={should_trigger_full_text}",
        )
        if not should_trigger_full_text:
            return None

        context_text, used_tokens, truncated = bundle.build_full_text_context(
            max_context_tokens=context_budget,
        )
        if not context_text or truncated:
            print(
                "ContextOrchestrator#full_text_gate:",
                f"scope={scope.value}",
                f"total_records={total_records}",
                f"estimated_full_text_tokens={estimated_full_text_tokens}",
                f"available_context_budget={context_budget}",
                f"effective_input_budget={effective_input_budget}",
                f"effective_output_budget={effective_output_budget}",
                f"effective_context_budget={effective_context_budget}",
                f"budget_policy={budget_policy}",
                f"capability_used={capability_used}",
                "triggered=False",
            )
            return None

        print(
            "ContextOrchestrator#context_mode: full_text_mode "
            f"(records={total_records}, token_used={used_tokens}, budget={context_budget})"
        )
        return ContextBuildResult(
            context_text=context_text,
            mode=ContextMode.FULL_TEXT,
            prompt_mode=prompt_mode,
            metadata={
                "token_used": used_tokens,
                "budget": context_budget,
                "records": total_records,
                "truncated": truncated,
                "estimated_full_text_tokens": estimated_full_text_tokens,
                "effective_input_budget": effective_input_budget,
                "effective_output_budget": effective_output_budget,
                "effective_context_budget": effective_context_budget,
                "full_text_budget_policy": budget_policy,
                "full_text_capability_used": capability_used,
            },
            standardized_question=question,
            answer_mode=answer_mode,
            results=[],
        )

    def build(
        self,
        query: str,
        bundle: FaissIndexBundle,
        top_k: int = 3,
        session_active_chunk_index: int | None = None,
    ) -> ContextBuildResult:
        """Build context by orchestrating search, relevance, and strategy decision."""
        if bundle.profile is None:
            print("Warn:FaissIndexBundle.answer: profile is not ready")

        standardized_question = self.question_standardizer.standardize(
            query=query,
            document_language=bundle.document_language,
        )
        scope_resolution: QuestionScopeResolution = self.question_scope_resolver.resolve(
            standardized_question
        )
        scope = scope_resolution.scope
        effective_top_k = (
            top_k
            if scope == QuestionScope.LOCAL
            else max(top_k, self.global_scope_min_top_k)
        )
        results = bundle.search(
            standardized_question.standardized_query,
            effective_top_k,
        )
        answer_mode: AnswerMode = self.relevance_evaluator.evaluate(results)
        print("FaissIndexBundle#ask standardized_question:", standardized_question)
        print("FaissIndexBundle#ask answer_mode:", answer_mode)
        print(
            "ContextOrchestrator#scope:",
            f"scope={scope.value}",
            f"method={scope_resolution.method}",
            f"lang={scope_resolution.query_language.value}",
            f"keyword={scope_resolution.matched_keyword}",
            f"similarity={scope_resolution.similarity}",
            f"requested_top_k={top_k}",
            f"effective_top_k={effective_top_k}",
        )

        if answer_mode.level == AnswerLevel.REJECT:
            print("FaissIndexBundle#context_mode: retrieval_mode (answer_reject)")
            return ContextBuildResult(
                context_text="",
                mode=ContextMode.RETRIEVAL,
                prompt_mode=PromptMode.RETRIEVAL,
                metadata={
                    "reason": "answer_reject",
                    "scope": scope.value,
                    "scope_method": scope_resolution.method,
                    "scope_language": scope_resolution.query_language.value,
                    "scope_keyword": scope_resolution.matched_keyword,
                    "scope_similarity": scope_resolution.similarity,
                    "requested_top_k": top_k,
                    "effective_top_k": effective_top_k,
                },
                standardized_question=standardized_question,
                answer_mode=answer_mode,
                results=results,
            )

        full_text_result = self._maybe_build_full_text_context(
            bundle=bundle,
            question=standardized_question,
            answer_mode=answer_mode,
            scope=scope,
        )
        if full_text_result is not None:
            return ContextBuildResult(
                context_text=full_text_result.context_text,
                mode=full_text_result.mode,
                prompt_mode=full_text_result.prompt_mode,
                metadata={
                    **full_text_result.metadata,
                    "scope": scope.value,
                    "scope_method": scope_resolution.method,
                    "scope_language": scope_resolution.query_language.value,
                    "scope_keyword": scope_resolution.matched_keyword,
                    "scope_similarity": scope_resolution.similarity,
                    "requested_top_k": top_k,
                    "effective_top_k": effective_top_k,
                },
                standardized_question=standardized_question,
                answer_mode=answer_mode,
                results=results,
            )

        near_chunk_threshold = self._resolve_near_chunk_threshold(
            answer_mode=answer_mode,
            results=results,
        )

        if scope == QuestionScope.LOCAL and results:
            best_result = results[0]
            best_chunk_index = self._extract_chunk_index(bundle, best_result)
            if (
                isinstance(session_active_chunk_index, int)
                and isinstance(best_chunk_index, int)
                and abs(best_chunk_index - session_active_chunk_index) <= near_chunk_threshold
            ):
                prompt_mode = PromptMode.LOCAL_READING
                context_budget = bundle.compute_available_context_budget(
                    question=standardized_question,
                    answer_mode=answer_mode,
                    prompt_mode=prompt_mode,
                )
                local_texts, used_tokens, truncated, used_radius = bundle.build_local_window_dynamic(
                    best_result,
                    max_context_tokens=context_budget,
                )
                if local_texts:
                    print(
                        "FaissIndexBundle#context_mode: local_window_mode "
                        f"(active={session_active_chunk_index}, best={best_chunk_index}, "
                        f"threshold={near_chunk_threshold}, radius={used_radius}, "
                        f"token_used={used_tokens}, budget={context_budget}, "
                        f"truncated={truncated})"
                    )
                    return ContextBuildResult(
                        context_text="\n".join(local_texts),
                        mode=ContextMode.LOCAL_WINDOW,
                        prompt_mode=prompt_mode,
                        metadata={
                            "scope": scope.value,
                            "scope_method": scope_resolution.method,
                            "scope_language": scope_resolution.query_language.value,
                            "scope_keyword": scope_resolution.matched_keyword,
                            "scope_similarity": scope_resolution.similarity,
                            "requested_top_k": top_k,
                            "effective_top_k": effective_top_k,
                            "active_chunk_index": session_active_chunk_index,
                            "best_chunk_index": best_chunk_index,
                            "threshold": near_chunk_threshold,
                            "used_radius": used_radius,
                            "token_used": used_tokens,
                            "budget": context_budget,
                            "truncated": truncated,
                        },
                        standardized_question=standardized_question,
                        answer_mode=answer_mode,
                        results=results,
                    )

        best_chunk_index = self._extract_chunk_index(bundle, results[0]) if results else None
        prompt_mode = PromptMode.RETRIEVAL
        context_budget = bundle.compute_available_context_budget(
            question=standardized_question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )
        print(
            "FaissIndexBundle#context_mode: retrieval_mode "
            f"(active={session_active_chunk_index}, best={best_chunk_index}, "
            f"threshold={near_chunk_threshold})"
        )
        context_results = results
        if scope == QuestionScope.GLOBAL:
            coverage_selection = self.global_coverage_context_builder.select_for_global_scope(
                bundle=bundle,
                results=results,
            )
            context_results = coverage_selection.selected_results
            print(
                "ContextOrchestrator#global_coverage:",
                f"raw_chunk_indices={coverage_selection.raw_chunk_indices}",
                f"selected_chunk_indices={coverage_selection.selected_chunk_indices}",
                f"evidence_count={len(context_results)}",
            )

        texts = self._extract_texts(context_results)
        context_text, used_tokens, truncated = bundle.join_texts_with_budget(
            texts,
            max_context_tokens=context_budget,
        )
        print(
            "FaissIndexBundle#context_budget: retrieval_mode "
            f"(token_used={used_tokens}, budget={context_budget}, truncated={truncated})"
        )
        return ContextBuildResult(
            context_text=context_text,
            mode=ContextMode.RETRIEVAL,
            prompt_mode=prompt_mode,
            metadata={
                "scope": scope.value,
                "scope_method": scope_resolution.method,
                "scope_language": scope_resolution.query_language.value,
                "scope_keyword": scope_resolution.matched_keyword,
                "scope_similarity": scope_resolution.similarity,
                "requested_top_k": top_k,
                "effective_top_k": effective_top_k,
                "active_chunk_index": session_active_chunk_index,
                "best_chunk_index": best_chunk_index,
                "threshold": near_chunk_threshold,
                "token_used": used_tokens,
                "budget": context_budget,
                "truncated": truncated,
                "global_coverage_evidence_count": len(context_results)
                if scope == QuestionScope.GLOBAL
                else None,
            },
            standardized_question=standardized_question,
            answer_mode=answer_mode,
            results=results,
        )
