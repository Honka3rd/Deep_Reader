#!/usr/bin/env python3
"""Regression tests for persisted analysis read/generate orchestration."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_orchestrator import (  # noqa: E402
    AnalysisInteractionOrchestrator,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class InMemoryAnalysisArtifactStore:
    def __init__(self) -> None:
        self.artifacts: dict[tuple[str, str, str], ReadingInteractionArtifact] = {}
        self.get_calls = 0
        self.save_calls = 0

    def get_analysis_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self.get_calls += 1
        return self.artifacts.get(self._key_from_request(request))

    def save_analysis_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self.save_calls += 1
        self.artifacts[self._key_from_artifact(artifact)] = artifact
        return artifact

    @staticmethod
    def _key_from_request(
        request: ReadingInteractionRequest,
    ) -> tuple[str, str, str]:
        target = request.target
        return (target.document_id, target.target_level, target.target_id)

    @staticmethod
    def _key_from_artifact(
        artifact: ReadingInteractionArtifact,
    ) -> tuple[str, str, str]:
        return (artifact.document_id, artifact.target_level, artifact.target_id)


def _target(content: str = "This section explains market incentives clearly.") -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="section",
        target_id="section-1",
        content=content,
        chapter_id="chapter-1",
        section_id="section-1",
    )


def _request(
    *,
    content: str = "This section explains market incentives clearly.",
    refresh: bool = False,
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content),
        interaction_type="analysis",
        refresh=refresh,
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="analysis_prompt_v1",
    )


def _completed_artifact() -> ReadingInteractionArtifact:
    return ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="analysis",
        status="completed",
        payload={
            "summary": "Cached summary",
            "reasoning": "Cached reasoning",
            "explanation": "Cached explanation",
        },
    )


def _insufficient_artifact() -> ReadingInteractionArtifact:
    return ReadingInteractionArtifact.from_target(
        target=_target("@@@ !!! ###"),
        interaction_type="analysis",
        status="insufficient_content",
        reason="target content is too short for analysis",
    )


def test_read_missing_artifact_does_not_generate_or_persist() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.read(_request())

    _assert(artifact.status == "not_generated", "missing read should report not_generated")
    _assert(generator_calls == [], "read path must not call generator")
    _assert(store.save_calls == 0, "read path must not persist a placeholder")


def test_read_existing_artifact_returns_persisted_result() -> None:
    store = InMemoryAnalysisArtifactStore()
    existing = _completed_artifact()
    store.save_analysis_artifact(existing)
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(lambda request: "{}"),
        artifact_store=store,
    )

    artifact = orchestrator.read(_request())

    _assert(artifact is existing, "read should return current persisted artifact")


def test_read_existing_insufficient_content_rechecks_current_target() -> None:
    store = InMemoryAnalysisArtifactStore()
    existing = _insufficient_artifact()
    store.save_analysis_artifact(existing)
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(lambda request: "{}"),
        artifact_store=store,
    )

    readable = orchestrator.read(_request())
    still_insufficient = orchestrator.read(_request(content="@@@ !!! ###"))

    _assert(
        readable.status == "not_generated",
        "readable targets should not stay masked by stale insufficient artifacts",
    )
    _assert(
        still_insufficient is existing,
        "still-insufficient targets should keep the persisted insufficient artifact",
    )


def test_generate_missing_artifact_persists_completed_result() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Generated summary",
            "reasoning": "Generated reasoning",
            "explanation": "Generated explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request())

    _assert(generator_calls == ["section-1"], "generate should call generator once")
    _assert(artifact.status == "completed", "valid generated analysis should complete")
    _assert(store.save_calls == 1, "completed analysis should be persisted")
    _assert(orchestrator.read(_request()).payload == artifact.payload, "read should see saved artifact")


def test_generate_persists_insufficient_content_without_generator_call() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request(content="@@@ !!! ###"))

    _assert(artifact.status == "insufficient_content", "noise should persist insufficient status")
    _assert(generator_calls == [], "insufficient content should skip generator")
    _assert(store.save_calls == 1, "insufficient-content status should be persisted")
    _assert(orchestrator.read(_request(content="@@@ !!! ###")).status == "insufficient_content", "read should see persisted insufficient-content artifact")


def test_generate_does_not_persist_invalid_output_failure() -> None:
    store = InMemoryAnalysisArtifactStore()
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(lambda request: "{not json"),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request())

    _assert(artifact.status == "generation_failed", "invalid JSON should fail generation")
    _assert(store.save_calls == 0, "generation failures should not be persisted")
    _assert(orchestrator.read(_request()).status == "not_generated", "failed generation should not create a readable artifact")


def test_generate_reuses_existing_artifact_unless_refresh_is_requested() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Refreshed summary",
            "reasoning": "Refreshed reasoning",
            "explanation": "Refreshed explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    existing = _completed_artifact()
    store.save_analysis_artifact(existing)
    store.save_calls = 0
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    reused = orchestrator.generate(_request())
    refreshed = orchestrator.generate(_request(refresh=True))

    _assert(reused is existing, "generate without refresh should reuse persisted artifact")
    _assert(generator_calls == ["section-1"], "only refresh should call generator")
    _assert(refreshed.payload["summary"] == "Refreshed summary", "refresh should persist new output")
    _assert(store.save_calls == 1, "only refresh should write a new artifact")


def test_generate_rechecks_existing_insufficient_content_artifact() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Recovered summary",
            "reasoning": "Recovered reasoning",
            "explanation": "Recovered explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    store.save_analysis_artifact(_insufficient_artifact())
    store.save_calls = 0
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    recovered = orchestrator.generate(_request())

    _assert(
        recovered.status == "completed",
        "generate should recheck persisted insufficient-content artifacts",
    )
    _assert(generator_calls == ["section-1"], "readable content should call generator")
    _assert(
        recovered.payload["summary"] == "Recovered summary",
        "rechecked analysis should replace the insufficient artifact",
    )
    _assert(store.save_calls == 1, "recovered analysis should be persisted")


def test_generate_rechecks_still_insufficient_artifact_without_generator_call() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }

    store = InMemoryAnalysisArtifactStore()
    store.save_analysis_artifact(_insufficient_artifact())
    store.save_calls = 0
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    still_insufficient = orchestrator.generate(_request(content="@@@ !!! ###"))

    _assert(
        still_insufficient.status == "insufficient_content",
        "still-insufficient content should remain insufficient",
    )
    _assert(generator_calls == [], "still-insufficient content should skip generator")
    _assert(store.save_calls == 1, "rechecked insufficient status should be persisted")


if __name__ == "__main__":
    test_read_missing_artifact_does_not_generate_or_persist()
    test_read_existing_artifact_returns_persisted_result()
    test_read_existing_insufficient_content_rechecks_current_target()
    test_generate_missing_artifact_persists_completed_result()
    test_generate_persists_insufficient_content_without_generator_call()
    test_generate_does_not_persist_invalid_output_failure()
    test_generate_reuses_existing_artifact_unless_refresh_is_requested()
    test_generate_rechecks_existing_insufficient_content_artifact()
    test_generate_rechecks_still_insufficient_artifact_without_generator_call()
    print("analysis interaction read/generate split tests passed")
