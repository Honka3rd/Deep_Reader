from __future__ import annotations

from dataclasses import replace
from typing import Any

from document_structure.document_artifact_repository import DocumentArtifactRepository
from section_tasks.analysis_interaction_orchestrator import AnalysisInteractionArtifactStore
from section_tasks.reading_interaction_common_artifact import (
    common_artifact_to_reading_interaction_artifact,
    reading_interaction_artifact_to_common_artifact,
)
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from shared.common_artifact_model import CommonArtifact
from shared.task_artifacts import DocumentTaskArtifacts


class DocumentReadingInteractionArtifactStore(AnalysisInteractionArtifactStore):
    """Persist current reading-interaction artifacts in document artifact metadata."""

    _METADATA_KEY = "reading_interaction_artifacts"

    def __init__(
        self,
        document_artifact_repository: DocumentArtifactRepository,
    ) -> None:
        self._document_artifact_repository = document_artifact_repository

    def get_analysis_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self._validate_request_type(request, expected_type="analysis")
        common_artifact = self._get_current_common_artifact(
            request=request,
            artifact_type="analysis",
        )
        if common_artifact is None:
            return None
        return common_artifact_to_reading_interaction_artifact(common_artifact)

    def save_analysis_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self._validate_artifact_type(artifact, expected_type="analysis")
        artifact_id = self._artifact_id_for(
            artifact_type=artifact.interaction_type,
            document_id=artifact.document_id,
            target_level=artifact.target_level,
            target_id=artifact.target_id,
        )
        metadata = dict(artifact.metadata)
        metadata["artifact_id"] = artifact_id
        artifact_to_save = ReadingInteractionArtifact(
            interaction_type=artifact.interaction_type,
            status=artifact.status,
            target_level=artifact.target_level,
            target_id=artifact.target_id,
            document_id=artifact.document_id,
            chapter_id=artifact.chapter_id,
            section_id=artifact.section_id,
            task_unit_id=artifact.task_unit_id,
            payload=dict(artifact.payload),
            metadata=metadata,
            reason=artifact.reason,
        )
        common_artifact = reading_interaction_artifact_to_common_artifact(
            artifact_to_save,
            artifact_id=artifact_id,
        )
        doc_name = self._doc_name_from_artifact(artifact_to_save)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
        document_metadata = dict(document_artifacts.metadata)
        interaction_metadata = self._read_interaction_metadata(document_metadata)
        artifact_type_payload = dict(
            interaction_metadata.get(artifact.interaction_type, {})
        )
        artifact_type_payload[common_artifact.artifact_id or ""] = (
            common_artifact.to_dict()
        )
        interaction_metadata[artifact.interaction_type] = artifact_type_payload
        document_metadata[self._METADATA_KEY] = interaction_metadata

        self._document_artifact_repository.update_document_artifacts(
            doc_name=doc_name,
            artifacts=replace(document_artifacts, metadata=document_metadata),
        )
        return artifact_to_save

    def _get_current_common_artifact(
        self,
        *,
        request: ReadingInteractionRequest,
        artifact_type: str,
    ) -> CommonArtifact | None:
        doc_name = self._doc_name_from_request(request)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None

        interaction_metadata = self._read_interaction_metadata(
            document_artifacts.metadata
        )
        artifact_id = self._artifact_id_for(
            artifact_type=artifact_type,
            document_id=request.target.document_id,
            target_level=request.target.target_level,
            target_id=request.target.target_id,
        )
        artifact_payload = interaction_metadata.get(artifact_type, {}).get(artifact_id)
        if not isinstance(artifact_payload, dict):
            return None
        return CommonArtifact.from_dict(artifact_payload)

    @classmethod
    def _read_interaction_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        raw_payload = metadata.get(cls._METADATA_KEY, {})
        if not isinstance(raw_payload, dict):
            return {}

        interaction_metadata: dict[str, dict[str, dict[str, Any]]] = {}
        for artifact_type, artifact_map in raw_payload.items():
            if not isinstance(artifact_map, dict):
                continue
            interaction_metadata[str(artifact_type)] = {
                str(artifact_id): dict(artifact_payload)
                for artifact_id, artifact_payload in artifact_map.items()
                if isinstance(artifact_payload, dict)
            }
        return interaction_metadata

    @staticmethod
    def _artifact_id_for(
        *,
        artifact_type: str,
        document_id: str,
        target_level: str,
        target_id: str,
    ) -> str:
        return "::".join(
            [
                artifact_type.strip(),
                document_id.strip(),
                target_level.strip(),
                target_id.strip(),
            ]
        )

    @staticmethod
    def _doc_name_from_request(request: ReadingInteractionRequest) -> str:
        doc_name = request.context_metadata.get("doc_name")
        if not isinstance(doc_name, str) or not doc_name.strip():
            raise ValueError("reading interaction request metadata requires doc_name")
        return doc_name.strip()

    @staticmethod
    def _doc_name_from_artifact(artifact: ReadingInteractionArtifact) -> str:
        context = artifact.metadata.get("context")
        if isinstance(context, dict):
            doc_name = context.get("doc_name")
            if isinstance(doc_name, str) and doc_name.strip():
                return doc_name.strip()
        raise ValueError("reading interaction artifact metadata requires context.doc_name")

    @staticmethod
    def _validate_request_type(
        request: ReadingInteractionRequest,
        *,
        expected_type: str,
    ) -> None:
        if request.interaction_type != expected_type:
            raise ValueError(
                f"artifact store requires interaction_type='{expected_type}'"
            )

    @staticmethod
    def _validate_artifact_type(
        artifact: ReadingInteractionArtifact,
        *,
        expected_type: str,
    ) -> None:
        if artifact.interaction_type != expected_type:
            raise ValueError(
                f"artifact store requires artifact type '{expected_type}'"
            )
