from __future__ import annotations

from shared.common_artifact_model import CommonArtifact, CommonArtifactTarget

from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact


def reading_interaction_artifact_to_common_artifact(
    artifact: ReadingInteractionArtifact,
    *,
    artifact_id: str | None = None,
    source_structure_version: int | None = None,
    source_hash: str | None = None,
) -> CommonArtifact:
    """Map a reading interaction result onto the single common artifact entity."""
    if source_structure_version is None:
        policy = artifact.metadata.get("interaction_policy")
        value = policy.get("source_structure_version") if isinstance(policy, dict) else None
        if type(value) is int and value > 0:
            source_structure_version = value
    return CommonArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact.interaction_type,
        status=artifact.status,
        target=CommonArtifactTarget(
            target_level=artifact.target_level,
            target_id=artifact.target_id,
            document_id=artifact.document_id,
            chapter_id=artifact.chapter_id,
            section_id=artifact.section_id,
            task_unit_id=artifact.task_unit_id,
            content_block_id=artifact.content_block_id,
        ),
        payload=dict(artifact.payload),
        metadata=dict(artifact.metadata),
        reason=artifact.reason,
        source_structure_version=source_structure_version,
        source_hash=source_hash,
    )


def common_artifact_to_reading_interaction_artifact(
    artifact: CommonArtifact,
) -> ReadingInteractionArtifact:
    """Return the service-level reading interaction artifact view for a common entity."""
    return ReadingInteractionArtifact(
        interaction_type=artifact.artifact_type,
        status=artifact.status,
        target_level=artifact.target.target_level,
        target_id=artifact.target.target_id,
        document_id=artifact.target.document_id,
        chapter_id=artifact.target.chapter_id,
        section_id=artifact.target.section_id,
        task_unit_id=artifact.target.task_unit_id,
        content_block_id=artifact.target.content_block_id,
        payload=dict(artifact.payload),
        metadata=dict(artifact.metadata),
        reason=artifact.reason,
    )
