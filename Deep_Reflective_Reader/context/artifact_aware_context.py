from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactContextSummary:
    """Compact lower-level artifact summary for secondary context assembly."""

    artifact_id: str
    artifact_type: str
    target_level: str
    target_id: str
    status: str = "completed"
    focus: str | None = None
    concepts: tuple[str, ...] = ()
    outcome: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactAwareContextResult:
    """Artifact-aware secondary context projection and provenance metadata."""

    context_text: str
    artifact_context_mode: str
    referenced_artifact_ids: list[str]
    referenced_artifact_types: list[str]
    referenced_artifact_target_levels: list[str]
    coverage_counts: dict[str, int]
    deduplication_hint_applied: bool
    abstraction_hint_applied: bool
    artifact_context_pruned_reason: str | None = None

    def to_metadata(self) -> dict[str, object]:
        """Return artifact provenance metadata without secondary context payload."""
        metadata: dict[str, object] = {
            "artifact_context_mode": self.artifact_context_mode,
            "referenced_artifact_ids": list(self.referenced_artifact_ids),
            "referenced_artifact_types": list(self.referenced_artifact_types),
            "referenced_artifact_target_levels": list(
                self.referenced_artifact_target_levels
            ),
            "coverage_counts": dict(self.coverage_counts),
            "deduplication_hint_applied": self.deduplication_hint_applied,
            "abstraction_hint_applied": self.abstraction_hint_applied,
        }
        if self.artifact_context_pruned_reason is not None:
            metadata["artifact_context_pruned_reason"] = (
                self.artifact_context_pruned_reason
            )
        return metadata


class ArtifactAwareContextBuilder:
    """Build compact lower-level artifact context without embedding payloads."""

    def build_secondary_context(
        self,
        artifacts: list[ArtifactContextSummary],
        *,
        max_artifacts: int | None = None,
        max_context_chars: int | None = None,
    ) -> ArtifactAwareContextResult:
        """Create a bounded secondary context from lower-level artifact summaries."""
        if max_artifacts is not None and max_artifacts < 0:
            raise ValueError("max_artifacts must be >= 0")
        if max_context_chars is not None and max_context_chars < 0:
            raise ValueError("max_context_chars must be >= 0")

        selected: list[ArtifactContextSummary] = []
        seen_artifact_ids: set[str] = set()
        coverage_counts: dict[str, int] = {}
        skipped_count = 0
        duplicate_count = 0

        for artifact in artifacts:
            artifact_id = artifact.artifact_id.strip()
            artifact_type = artifact.artifact_type.strip()
            target_level = artifact.target_level.strip()
            target_id = artifact.target_id.strip()
            status = artifact.status.strip()

            if not artifact_id or not artifact_type or not target_level or not target_id:
                skipped_count += 1
                continue
            if status == "insufficient_content":
                skipped_count += 1
                continue
            if artifact_id in seen_artifact_ids:
                duplicate_count += 1
                continue
            if max_artifacts is not None and len(selected) >= max_artifacts:
                skipped_count += 1
                continue

            seen_artifact_ids.add(artifact_id)
            selected.append(artifact)
            coverage_counts[target_level] = coverage_counts.get(target_level, 0) + 1

        if not selected:
            reason = None
            if skipped_count or duplicate_count:
                reason = self._format_pruned_reason(
                    skipped_count=skipped_count,
                    duplicate_count=duplicate_count,
                )
            return ArtifactAwareContextResult(
                context_text="",
                artifact_context_mode="none",
                referenced_artifact_ids=[],
                referenced_artifact_types=[],
                referenced_artifact_target_levels=[],
                coverage_counts={},
                deduplication_hint_applied=False,
                abstraction_hint_applied=False,
                artifact_context_pruned_reason=reason,
            )

        lines: list[str] = ["Secondary lower-level artifact context:"]
        referenced_artifact_ids: list[str] = []
        referenced_artifact_types: list[str] = []
        referenced_artifact_target_levels: list[str] = []
        has_deduplication_signal = False
        has_abstraction_signal = False
        budget_pruned_count = 0

        for index, artifact in enumerate(selected, start=1):
            artifact_id = artifact.artifact_id.strip()
            artifact_type = artifact.artifact_type.strip()
            target_level = artifact.target_level.strip()
            target_id = artifact.target_id.strip()
            status = artifact.status.strip()

            referenced_artifact_ids.append(artifact_id)
            referenced_artifact_types.append(artifact_type)
            referenced_artifact_target_levels.append(target_level)

            concepts = ", ".join(
                concept.strip()
                for concept in artifact.concepts
                if concept.strip()
            )
            focus = (artifact.focus or "").strip()
            outcome = (artifact.outcome or "").strip()

            parts = [
                f"{index}. {artifact_type}",
                f"target={target_level}:{target_id}",
                f"status={status}",
            ]
            if focus:
                parts.append(f"focus={focus}")
            if concepts:
                parts.append(f"concepts={concepts}")
            if outcome:
                parts.append(f"outcome={outcome}")

            lines.append("; ".join(parts))

            if artifact_type == "quiz" or concepts:
                has_deduplication_signal = True
            if artifact_type in {"analysis", "critical_thinking_session"} or focus or outcome:
                has_abstraction_signal = True

        if max_context_chars is not None:
            budgeted_lines = self._apply_context_char_budget(
                lines=lines,
                max_context_chars=max_context_chars,
            )
            kept_artifact_count = max(0, len(budgeted_lines) - 1)
            budget_pruned_count = max(0, len(selected) - kept_artifact_count)
            lines = budgeted_lines
            if kept_artifact_count < len(referenced_artifact_ids):
                referenced_artifact_ids = referenced_artifact_ids[:kept_artifact_count]
                referenced_artifact_types = referenced_artifact_types[:kept_artifact_count]
                referenced_artifact_target_levels = referenced_artifact_target_levels[
                    :kept_artifact_count
                ]
                coverage_counts = self._coverage_counts_for_selected(
                    selected[:kept_artifact_count]
                )

        pruned_reason = None
        if skipped_count or duplicate_count or budget_pruned_count:
            pruned_reason = self._format_pruned_reason(
                skipped_count=skipped_count,
                duplicate_count=duplicate_count,
                budget_pruned_count=budget_pruned_count,
            )

        return ArtifactAwareContextResult(
            context_text="\n".join(lines) if referenced_artifact_ids else "",
            artifact_context_mode=("pruned" if pruned_reason else "referenced"),
            referenced_artifact_ids=referenced_artifact_ids,
            referenced_artifact_types=referenced_artifact_types,
            referenced_artifact_target_levels=referenced_artifact_target_levels,
            coverage_counts=coverage_counts,
            deduplication_hint_applied=has_deduplication_signal,
            abstraction_hint_applied=has_abstraction_signal,
            artifact_context_pruned_reason=pruned_reason,
        )

    @staticmethod
    def _format_pruned_reason(
        *,
        skipped_count: int,
        duplicate_count: int,
        budget_pruned_count: int = 0,
    ) -> str | None:
        reasons: list[str] = []
        if skipped_count:
            reasons.append(f"skipped={skipped_count}")
        if duplicate_count:
            reasons.append(f"duplicates={duplicate_count}")
        if budget_pruned_count:
            reasons.append(f"budget_pruned={budget_pruned_count}")
        return ", ".join(reasons) if reasons else None

    @staticmethod
    def _apply_context_char_budget(
        *,
        lines: list[str],
        max_context_chars: int,
    ) -> list[str]:
        if max_context_chars <= 0 or not lines:
            return []

        selected_lines: list[str] = []
        used_chars = 0
        for line in lines:
            separator_chars = 1 if selected_lines else 0
            projected_chars = used_chars + separator_chars + len(line)
            if projected_chars > max_context_chars:
                break
            selected_lines.append(line)
            used_chars = projected_chars
        return selected_lines

    @staticmethod
    def _coverage_counts_for_selected(
        selected: list[ArtifactContextSummary],
    ) -> dict[str, int]:
        coverage_counts: dict[str, int] = {}
        for artifact in selected:
            target_level = artifact.target_level.strip()
            if target_level:
                coverage_counts[target_level] = coverage_counts.get(target_level, 0) + 1
        return coverage_counts
