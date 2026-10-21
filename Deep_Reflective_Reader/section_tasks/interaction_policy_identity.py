"""Server-owned cache identity and read-only stale projections."""

from dataclasses import replace
import hashlib

from document_structure.source_mapped_text import POLICY_VERSION as SOURCE_VIEW_VERSION
from section_tasks.artifact_validity import SEMANTIC_GUARD_VERSION
from section_tasks.interaction_output_language import request_language, OUTPUT_LANGUAGE_POLICY_VERSION
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact, ReadingInteractionRequest,
)


POLICY_METADATA_KEY = "interaction_policy"


def prompt_version(interaction_type: str) -> str:
    # Imported at call time to keep service/prompt policy dependencies acyclic.
    from section_tasks.analysis_interaction_llm_generator import ANALYSIS_INTERACTION_PROMPT_VERSION
    from section_tasks.quiz_interaction_llm_generator import QUIZ_INTERACTION_PROMPT_VERSION
    from section_tasks.critical_thinking_llm_generator import CRITICAL_THINKING_PROMPT_VERSION
    return {
        "analysis": ANALYSIS_INTERACTION_PROMPT_VERSION,
        "quiz": QUIZ_INTERACTION_PROMPT_VERSION,
        "critical_thinking_session": CRITICAL_THINKING_PROMPT_VERSION,
        "summary": "summary_interaction_prompt_v1",
    }[interaction_type]


def effective_policy(request: ReadingInteractionRequest) -> dict[str, object]:
    target = request.target
    policy = {
        "schema": "interaction_policy_v1",
        "interaction_type": request.interaction_type,
        "prompt": prompt_version(request.interaction_type),
        "language_policy": OUTPUT_LANGUAGE_POLICY_VERSION,
        "language": request_language(request),
        "guard": SEMANTIC_GUARD_VERSION if request.interaction_type == "analysis" else "quantity-v1",
        "source_policy": SOURCE_VIEW_VERSION,
        "source_sha256": hashlib.sha256(target.content.encode("utf-8")).hexdigest(),
        "target": [target.document_id, target.target_level, target.target_id,
                   target.chapter_id, target.section_id, target.task_unit_id,
                   target.content_block_id],
    }
    if target.source_structure_version is not None:
        policy["source_structure_version"] = target.source_structure_version
    if target.reading_revision or target.source_needs_review:
        policy.update(reading_revision=target.reading_revision, source_needs_review=target.source_needs_review)
    return policy


def policy_metadata(request: ReadingInteractionRequest) -> dict[str, object]:
    return {POLICY_METADATA_KEY: effective_policy(request),
            "prompt_instruction_version": prompt_version(request.interaction_type)}


def server_request(request: ReadingInteractionRequest) -> ReadingInteractionRequest:
    return replace(request, prompt_instruction_version=prompt_version(request.interaction_type))


def policy_matches(artifact: ReadingInteractionArtifact, request: ReadingInteractionRequest) -> bool:
    return artifact.metadata.get(POLICY_METADATA_KEY) == effective_policy(request)


def stale_projection(artifact: ReadingInteractionArtifact, request: ReadingInteractionRequest) -> ReadingInteractionArtifact:
    """Never alter stored history; retain session payload and identifiers."""
    if policy_matches(artifact, request):
        return artifact
    payload = artifact.payload if artifact.interaction_type == "critical_thinking_session" else {}
    return replace(artifact, status="stale_target", reason="interaction_policy_stale", payload=payload)
