from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from document_structure.document_artifact_repository import DocumentArtifactRepository
from section_tasks.ancestor_summary_context import (
    AncestorSummaryArtifact,
    AncestorSummaryReader,
    AncestorSummaryRequirement,
)
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


@dataclass(frozen=True)
class DocumentCriticalThinkingSessionStoreResult:
    """Saved critical-thinking session result."""

    session_id: str
    artifact: ReadingInteractionArtifact


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
        return self._save_current_artifact(artifact, expected_type="analysis")

    def get_quiz_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self._validate_request_type(request, expected_type="quiz")
        common_artifact = self._get_current_common_artifact(
            request=request,
            artifact_type="quiz",
        )
        if common_artifact is None:
            return None
        return common_artifact_to_reading_interaction_artifact(common_artifact)

    def save_quiz_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        return self._save_current_artifact(artifact, expected_type="quiz")

    def get_critical_thinking_session(
        self,
        request: ReadingInteractionRequest,
        session_id: str,
    ) -> ReadingInteractionArtifact | None:
        self._validate_request_type(
            request,
            expected_type="critical_thinking_session",
        )
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("critical-thinking session_id is required")

        doc_name = self._doc_name_from_request(request)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None

        interaction_metadata = self._read_interaction_metadata(
            document_artifacts.metadata
        )
        artifact_payload = interaction_metadata.get(
            "critical_thinking_session",
            {},
        ).get(normalized_session_id)
        if not isinstance(artifact_payload, dict):
            return None

        common_artifact = CommonArtifact.from_dict(artifact_payload)
        artifact = common_artifact_to_reading_interaction_artifact(common_artifact)
        if not self._artifact_matches_request(artifact, request):
            return None
        return artifact

    def get_current_critical_thinking_session(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        """Return the latest persisted critical-thinking session for the request target."""
        self._validate_request_type(
            request,
            expected_type="critical_thinking_session",
        )
        doc_name = self._doc_name_from_request(request)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None

        interaction_metadata = self._read_interaction_metadata(
            document_artifacts.metadata
        )
        session_payloads = interaction_metadata.get(
            "critical_thinking_session",
            {},
        )
        matched_artifact: ReadingInteractionArtifact | None = None
        for artifact_payload in session_payloads.values():
            if not isinstance(artifact_payload, dict):
                continue
            common_artifact = CommonArtifact.from_dict(artifact_payload)
            artifact = common_artifact_to_reading_interaction_artifact(common_artifact)
            if self._artifact_matches_request(artifact, request):
                matched_artifact = artifact
        return matched_artifact

    def save_critical_thinking_session(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> DocumentCriticalThinkingSessionStoreResult:
        self._validate_artifact_type(
            artifact,
            expected_type="critical_thinking_session",
        )
        session_id = self._session_id_from_artifact(artifact)
        metadata = dict(artifact.metadata)
        metadata["session_id"] = session_id
        artifact_to_save = ReadingInteractionArtifact(
            interaction_type=artifact.interaction_type,
            status=artifact.status,
            target_level=artifact.target_level,
            target_id=artifact.target_id,
            document_id=artifact.document_id,
            chapter_id=artifact.chapter_id,
            section_id=artifact.section_id,
            task_unit_id=artifact.task_unit_id,
            content_block_id=artifact.content_block_id,
            payload=dict(artifact.payload),
            metadata=metadata,
            reason=artifact.reason,
        )
        self._save_common_artifact(
            artifact=artifact_to_save,
            artifact_id=session_id,
        )
        return DocumentCriticalThinkingSessionStoreResult(
            session_id=session_id,
            artifact=artifact_to_save,
        )

    def _save_current_artifact(
        self,
        artifact: ReadingInteractionArtifact,
        *,
        expected_type: str,
    ) -> ReadingInteractionArtifact:
        self._validate_artifact_type(artifact, expected_type=expected_type)
        artifact_id = self._artifact_id_for(
            artifact_type=artifact.interaction_type,
            document_id=artifact.document_id,
            target_level=artifact.target_level,
            target_id=artifact.target_id,
            content_block_id=artifact.content_block_id,
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
            content_block_id=artifact.content_block_id,
            payload=dict(artifact.payload),
            metadata=metadata,
            reason=artifact.reason,
        )
        self._save_common_artifact(
            artifact=artifact_to_save,
            artifact_id=artifact_id,
        )
        return artifact_to_save

    def _save_common_artifact(
        self,
        *,
        artifact: ReadingInteractionArtifact,
        artifact_id: str,
    ) -> None:
        common_artifact = reading_interaction_artifact_to_common_artifact(
            artifact,
            artifact_id=artifact_id,
        )
        doc_name = self._doc_name_from_artifact(artifact)
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
            content_block_id=request.target.content_block_id,
        )
        artifact_payload = interaction_metadata.get(artifact_type, {}).get(artifact_id)
        if not isinstance(artifact_payload, dict):
            return None
        return CommonArtifact.from_dict(artifact_payload)

    @staticmethod
    def _artifact_matches_request(
        artifact: ReadingInteractionArtifact,
        request: ReadingInteractionRequest,
    ) -> bool:
        target = request.target
        return (
            artifact.interaction_type == request.interaction_type
            and artifact.document_id == target.document_id
            and artifact.target_level == target.target_level
            and artifact.target_id == target.target_id
            and artifact.chapter_id == target.chapter_id
            and artifact.section_id == target.section_id
            and artifact.task_unit_id == target.task_unit_id
            and artifact.content_block_id == target.content_block_id
        )

    @classmethod
    def _session_id_from_artifact(
        cls,
        artifact: ReadingInteractionArtifact,
    ) -> str:
        raw_session_id = artifact.metadata.get("session_id")
        if isinstance(raw_session_id, str) and raw_session_id.strip():
            return raw_session_id.strip()
        session_id_parts = [
            artifact.interaction_type,
            artifact.document_id.strip(),
            artifact.target_level.strip(),
            artifact.target_id.strip(),
        ]
        content_block_id = (
            None
            if artifact.content_block_id is None
            else artifact.content_block_id.strip()
        )
        if content_block_id:
            session_id_parts.extend(["content_block", content_block_id])
        session_id_parts.append(uuid4().hex)
        return "::".join(session_id_parts)

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
        content_block_id: str | None = None,
    ) -> str:
        artifact_id_parts = [
            artifact_type.strip(),
            document_id.strip(),
            target_level.strip(),
            target_id.strip(),
        ]
        normalized_content_block_id = (
            None if content_block_id is None else content_block_id.strip()
        )
        if normalized_content_block_id:
            artifact_id_parts.extend(["content_block", normalized_content_block_id])
        return "::".join(artifact_id_parts)

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


class DocumentAncestorSummaryArtifactStore(AncestorSummaryReader):
    """Persist ancestor summary artifacts in document-level derived metadata."""

    _METADATA_KEY = "ancestor_summary_artifacts"

    def __init__(
        self,
        document_artifact_repository: DocumentArtifactRepository,
    ) -> None:
        self._document_artifact_repository = document_artifact_repository

    def get_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact | None:
        doc_name = self._doc_name_from_requirement(requirement)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None

        summary_metadata = self._read_summary_metadata(document_artifacts.metadata)
        artifact_payload = summary_metadata.get(self._artifact_id_for(requirement))
        if not isinstance(artifact_payload, dict):
            return None
        return AncestorSummaryArtifact.from_dict(artifact_payload)

    def save_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
        summary: AncestorSummaryArtifact,
    ) -> AncestorSummaryArtifact:
        artifact_id = self._artifact_id_for(requirement)
        summary_to_save = AncestorSummaryArtifact(
            artifact_id=artifact_id,
            target=summary.target,
            summary=summary.summary,
            schema_version=summary.schema_version,
            prompt_instruction_version=summary.prompt_instruction_version,
            source_structure_version=summary.source_structure_version,
            source_hash=summary.source_hash,
            document_language=summary.document_language,
            generated_at=summary.generated_at,
            metadata=dict(summary.metadata),
        )
        doc_name = self._doc_name_from_requirement(requirement)
        document = self._document_artifact_repository.load_document(doc_name)
        document_artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
        document_metadata = dict(document_artifacts.metadata)
        summary_metadata = self._read_summary_metadata(document_metadata)
        summary_metadata[artifact_id] = summary_to_save.to_dict()
        document_metadata[self._METADATA_KEY] = summary_metadata

        self._document_artifact_repository.update_document_artifacts(
            doc_name=doc_name,
            artifacts=replace(document_artifacts, metadata=document_metadata),
        )
        return summary_to_save

    @classmethod
    def _artifact_id_for(
        cls,
        requirement: AncestorSummaryRequirement,
    ) -> str:
        target = requirement.target
        return "::".join(
            [
                "ancestor_summary",
                target.document_id.strip(),
                target.target_level.strip(),
                target.target_id.strip(),
            ]
        )

    @classmethod
    def _read_summary_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        raw_payload = metadata.get(cls._METADATA_KEY, {})
        if not isinstance(raw_payload, dict):
            return {}
        return {
            str(artifact_id): dict(artifact_payload)
            for artifact_id, artifact_payload in raw_payload.items()
            if isinstance(artifact_payload, dict)
        }

    @staticmethod
    def _doc_name_from_requirement(requirement: AncestorSummaryRequirement) -> str:
        if requirement.doc_name:
            return requirement.doc_name
        return requirement.target.document_id
