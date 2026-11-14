"""Explicit generation snapshots for artifact repository regression fixtures."""
from dataclasses import replace
from document_structure.generated_artifact_source import BINDING_KEY, capture_generation_source


def bound(artifact, document, level, target_id):
    return replace(artifact, metadata={**(artifact.metadata or {}),
        BINDING_KEY: capture_generation_source(document, level, target_id)})


def bound_fields(artifacts, document, level, target_id):
    return replace(artifacts, summary=(bound(artifacts.summary, document, level, target_id)
        if artifacts.summary else None), quiz=(bound(artifacts.quiz, document, level, target_id)
        if artifacts.quiz else None))


def bound_layout(document, metadata):
    from document_structure.task_layout_source import LAYOUT_SOURCE_KEY, capture_layout_source
    return {**metadata, LAYOUT_SOURCE_KEY: capture_layout_source(document)}
