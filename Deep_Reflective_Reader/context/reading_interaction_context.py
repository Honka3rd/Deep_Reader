from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from context.token_budget_manager import TokenBudgetManager
from llm.llm_model_capabilities import LLMModelCapabilities


CONTEXT_MODE_FULL_TARGET = "full_target"
CONTEXT_MODE_SEMANTIC_COMPACT = "semantic_compact"
COMPACTION_REASON_TARGET_EXCEEDS_BUDGET = "target_exceeds_effective_context_budget"


class ResolvedReadingTargetLike(Protocol):
    """Resolved hierarchy target shape required by the context layer."""

    document_id: str
    document_title: str
    target_level: str
    target_id: str
    content: str
    chapter_id: str | None
    section_id: str | None
    task_unit_id: str | None


@dataclass(frozen=True)
class ReadingInteractionContextResult:
    """Primary source context selected for a reading interaction."""

    context_text: str
    context_mode: str
    token_estimate: int
    used_context_tokens: int
    effective_context_budget: int
    evidence_ids: list[str]
    target_level: str
    target_id: str
    model_capability_source: str
    compaction_reason: str | None = None
    truncated: bool = False

    def to_metadata(self) -> dict[str, object]:
        """Return persistence-safe provenance metadata without source text."""
        metadata: dict[str, object] = {
            "context_mode": self.context_mode,
            "token_estimate": self.token_estimate,
            "used_context_tokens": self.used_context_tokens,
            "effective_context_budget": self.effective_context_budget,
            "evidence_ids": list(self.evidence_ids),
            "target_level": self.target_level,
            "target_id": self.target_id,
            "model_capability_source": self.model_capability_source,
            "truncated": self.truncated,
        }
        if self.compaction_reason:
            metadata["compaction_reason"] = self.compaction_reason
        return metadata


class ReadingInteractionContextBuilder:
    """Build full or compact primary source context for reading interactions."""

    def __init__(
        self,
        *,
        token_budget_manager: TokenBudgetManager | None = None,
        default_max_context_tokens: int = 4000,
        default_reserved_output_tokens: int = 800,
    ) -> None:
        if default_max_context_tokens <= 0:
            raise ValueError("default_max_context_tokens must be > 0")
        if default_reserved_output_tokens < 0:
            raise ValueError("default_reserved_output_tokens must be >= 0")
        self.token_budget_manager = token_budget_manager or TokenBudgetManager(
            prompt_assembler=None
        )
        self.default_max_context_tokens = default_max_context_tokens
        self.default_reserved_output_tokens = default_reserved_output_tokens

    def build_context(
        self,
        target: ResolvedReadingTargetLike,
        *,
        fixed_prompt_instruction: str = "",
        max_context_tokens: int | None = None,
        model_capabilities: LLMModelCapabilities | None = None,
        reserved_output_tokens: int | None = None,
    ) -> ReadingInteractionContextResult:
        """Select full target context when it fits, otherwise compact deterministically."""
        self._validate_target(target)
        effective_budget = self._compute_effective_budget(
            fixed_prompt_instruction=fixed_prompt_instruction,
            max_context_tokens=max_context_tokens,
            model_capabilities=model_capabilities,
            reserved_output_tokens=reserved_output_tokens,
        )
        target_content = target.content.strip()
        target_token_estimate = self.token_budget_manager.estimate_tokens(target_content)
        model_capability_source = (
            model_capabilities.model_name
            if model_capabilities is not None
            else "configured_budget"
        )

        if target_token_estimate <= effective_budget:
            return ReadingInteractionContextResult(
                context_text=target_content,
                context_mode=CONTEXT_MODE_FULL_TARGET,
                token_estimate=target_token_estimate,
                used_context_tokens=target_token_estimate,
                effective_context_budget=effective_budget,
                evidence_ids=[self._target_evidence_id(target)],
                target_level=target.target_level,
                target_id=target.target_id,
                model_capability_source=model_capability_source,
            )

        return self._build_compact_context(
            target=target,
            target_content=target_content,
            target_token_estimate=target_token_estimate,
            effective_budget=effective_budget,
            model_capability_source=model_capability_source,
        )

    def _compute_effective_budget(
        self,
        *,
        fixed_prompt_instruction: str,
        max_context_tokens: int | None,
        model_capabilities: LLMModelCapabilities | None,
        reserved_output_tokens: int | None,
    ) -> int:
        configured_context_budget = (
            self.default_max_context_tokens
            if max_context_tokens is None
            else max_context_tokens
        )
        if configured_context_budget < 0:
            raise ValueError("max_context_tokens must be >= 0")
        if model_capabilities is None:
            return configured_context_budget

        output_reserve = (
            self.default_reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        if output_reserve < 0:
            raise ValueError("reserved_output_tokens must be >= 0")
        instruction_tokens = self.token_budget_manager.estimate_tokens(
            fixed_prompt_instruction
        )
        capability_budget = (
            model_capabilities.max_input_tokens - instruction_tokens - output_reserve
        )
        return max(0, min(configured_context_budget, capability_budget))

    def _build_compact_context(
        self,
        *,
        target: ResolvedReadingTargetLike,
        target_content: str,
        target_token_estimate: int,
        effective_budget: int,
        model_capability_source: str,
    ) -> ReadingInteractionContextResult:
        if effective_budget <= 0:
            return ReadingInteractionContextResult(
                context_text="",
                context_mode=CONTEXT_MODE_SEMANTIC_COMPACT,
                token_estimate=target_token_estimate,
                used_context_tokens=0,
                effective_context_budget=effective_budget,
                evidence_ids=[],
                target_level=target.target_level,
                target_id=target.target_id,
                model_capability_source=model_capability_source,
                compaction_reason=COMPACTION_REASON_TARGET_EXCEEDS_BUDGET,
                truncated=True,
            )

        chunks = self._split_semantic_chunks(target_content, effective_budget)
        ordered_indices = self._coverage_order(len(chunks))
        selected_texts: list[str] = []
        selected_evidence_ids: list[str] = []

        for chunk_index in ordered_indices:
            evidence_id = self._chunk_evidence_id(target, chunk_index)
            selected_texts.append(f"[{evidence_id}] {chunks[chunk_index]}")
            selected_evidence_ids.append(evidence_id)

        context_text, used_tokens, truncated = (
            self.token_budget_manager.join_texts_with_budget(
                selected_texts,
                default_max_context_tokens=effective_budget,
                max_context_tokens=effective_budget,
            )
        )
        selected_count = len(context_text.split("\n")) if context_text else 0
        return ReadingInteractionContextResult(
            context_text=context_text,
            context_mode=CONTEXT_MODE_SEMANTIC_COMPACT,
            token_estimate=target_token_estimate,
            used_context_tokens=used_tokens,
            effective_context_budget=effective_budget,
            evidence_ids=selected_evidence_ids[:selected_count],
            target_level=target.target_level,
            target_id=target.target_id,
            model_capability_source=model_capability_source,
            compaction_reason=COMPACTION_REASON_TARGET_EXCEEDS_BUDGET,
            truncated=truncated or selected_count < len(chunks),
        )

    def _split_semantic_chunks(self, content: str, effective_budget: int) -> list[str]:
        chunk_budget = max(40, effective_budget // 3)
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", content)
            if paragraph.strip()
        ]
        if not paragraphs:
            return [content]

        chunks: list[str] = []
        for paragraph in paragraphs:
            if self.token_budget_manager.estimate_tokens(paragraph) <= chunk_budget:
                chunks.append(paragraph)
                continue

            current: list[str] = []
            current_tokens = 0
            for sentence in self._split_sentences(paragraph):
                sentence_tokens = self.token_budget_manager.estimate_tokens(sentence)
                if current and current_tokens + sentence_tokens > chunk_budget:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                if sentence_tokens > chunk_budget:
                    clipped = self.token_budget_manager.truncate_text_to_token_budget(
                        sentence,
                        chunk_budget,
                    )
                    if clipped:
                        chunks.append(clipped)
                    continue
                current.append(sentence)
                current_tokens += sentence_tokens
            if current:
                chunks.append(" ".join(current).strip())
        return chunks or [content]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?。！？;；])\s+", text)
            if part.strip()
        ]

    @staticmethod
    def _coverage_order(chunk_count: int) -> list[int]:
        if chunk_count <= 0:
            return []
        priority = [0, chunk_count // 2, chunk_count - 1]
        seen: set[int] = set()
        ordered: list[int] = []
        for index in priority + list(range(chunk_count)):
            if index not in seen:
                ordered.append(index)
                seen.add(index)
        return ordered

    @staticmethod
    def _validate_target(target: ResolvedReadingTargetLike) -> None:
        if not target.target_level.strip():
            raise ValueError("resolved target level is required")
        if not target.target_id.strip():
            raise ValueError("resolved target id is required")
        if not target.content.strip():
            raise ValueError("resolved target content is required")

    @staticmethod
    def _target_evidence_id(target: ResolvedReadingTargetLike) -> str:
        return f"{target.target_level}:{target.target_id}"

    @staticmethod
    def _chunk_evidence_id(target: ResolvedReadingTargetLike, chunk_index: int) -> str:
        return f"{target.target_level}:{target.target_id}#chunk-{chunk_index + 1}"
