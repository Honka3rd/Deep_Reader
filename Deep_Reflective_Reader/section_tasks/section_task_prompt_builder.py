from enum import StrEnum

from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext


class SectionTaskType(StrEnum):
    """Supported section-task prompt types."""

    SUMMARY = "summary"
    QUIZ = "quiz"


class SectionTaskPromptBuilder:
    """Build task prompts for section-based chapter services."""

    def build(
        self,
        *,
        task_type: SectionTaskType | str,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
    ) -> str:
        """Build one task prompt from normalized task context and optional profile."""
        resolved_task_type = self._resolve_task_type(task_type)
        task_instruction = self._build_task_instruction(resolved_task_type)
        profile_block = self._build_profile_block(document_profile)
        language_instruction = self._build_language_instruction(document_profile)
        topic_instruction = self._build_topic_instruction(document_profile)
        container_title_line = self._build_container_title_line(context)

        return (
            "You are working on one section-based reading task.\n"
            "Use section content as the primary evidence.\n"
            "Document profile is optional background only.\n\n"
            f"Task Type: {resolved_task_type.value}\n"
            f"{task_instruction}\n"
            f"{language_instruction}\n"
            f"{topic_instruction}\n\n"
            f"{profile_block}\n"
            "Section Context:\n"
            f"- Document Title: {self._display_document_title(context)}\n"
            f"{container_title_line}"
            f"- Section ID: {context.section_id}\n"
            f"- Section Title: {self._display_section_title(context)}\n"
            f"- Section Index: {context.section_index}\n"
            "\n"
            "Section Content:\n"
            f"{context.section_content}\n"
        )

    @staticmethod
    def _resolve_task_type(task_type: SectionTaskType | str) -> SectionTaskType:
        """Resolve external task type values to supported enum."""
        if isinstance(task_type, SectionTaskType):
            return task_type
        normalized = task_type.strip().lower()
        try:
            return SectionTaskType(normalized)
        except ValueError as error:
            supported = ", ".join(value.value for value in SectionTaskType)
            raise ValueError(
                f"unsupported section task type: {task_type!r}. supported: {supported}"
            ) from error

    @staticmethod
    def _build_task_instruction(task_type: SectionTaskType) -> str:
        """Return task-specific prompt instructions."""
        if task_type == SectionTaskType.SUMMARY:
            return (
                "Task:\n"
                "- Summarize only this section.\n"
                "- Keep output to 1-3 short paragraphs.\n"
                "- Do not include facts outside this section."
            )
        return (
            "Task:\n"
            "- Create 3 to 5 short-answer reading-comprehension questions.\n"
            "- Provide concise standard answers.\n"
            "- Use only this section.\n"
            "- Output in plain text with Q1/A1, Q2/A2 format."
        )

    def _build_profile_block(self, document_profile: DocumentProfile | None) -> str:
        """Return compact profile background block."""
        topic = self._extract_profile_topic(document_profile)
        language = self._extract_profile_language(document_profile)
        summary = self._extract_profile_summary(document_profile)

        if topic is None and language is None and summary is None:
            return "Document Profile: None"

        return (
            "Document Profile (background only):\n"
            f"- Topic: {topic or 'unknown'}\n"
            f"- Language: {language or 'unknown'}\n"
            f"- Summary: {summary or 'none'}"
        )

    def _build_language_instruction(self, document_profile: DocumentProfile | None) -> str:
        """Return output language instruction based on profile language signal."""
        language = self._extract_profile_language(document_profile)
        normalized = (language or "").strip().lower()
        if normalized.startswith("zh"):
            return "Output Language: Chinese"
        if normalized.startswith("en"):
            return "Output Language: English"
        return "Output Language: Follow the section content language"

    def _build_topic_instruction(self, document_profile: DocumentProfile | None) -> str:
        """Return light topic-aware instruction without overriding section evidence."""
        topic = self._extract_profile_topic(document_profile)
        normalized = (topic or "").strip().lower()
        if not normalized:
            return "Topic Guidance: None"
        if any(token in normalized for token in ("literary", "novel", "fiction", "literature")):
            return "Topic Guidance: emphasize characters, relationships, and key events."
        if any(token in normalized for token in ("financial", "report", "finance", "earnings")):
            return "Topic Guidance: emphasize entities, periods, and concrete financial points."
        if any(token in normalized for token in ("technical", "documentation", "manual", "api")):
            return "Topic Guidance: emphasize definitions, procedures, and constraints."
        if any(token in normalized for token in ("biography", "history", "historical")):
            return "Topic Guidance: emphasize people, timeline, and turning points."
        return f"Topic Guidance: align style with topic '{topic}'."

    @staticmethod
    def _extract_profile_topic(document_profile: DocumentProfile | None) -> str | None:
        """Extract normalized topic from document profile."""
        if document_profile is None:
            return None
        text_value = str(document_profile.topic).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _extract_profile_language(document_profile: DocumentProfile | None) -> str | None:
        """Extract normalized language from document profile."""
        if document_profile is None:
            return None
        text_value = str(document_profile.document_language).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _extract_profile_summary(document_profile: DocumentProfile | None) -> str | None:
        """Extract normalized summary from document profile."""
        if document_profile is None:
            return None
        text_value = str(document_profile.summary).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _build_container_title_line(context: SectionTaskContext) -> str:
        """Build optional container-title line from normalized context."""
        if context.container_title is None:
            return ""
        return f"- Container Title: {context.container_title}\n"

    @staticmethod
    def _display_document_title(context: SectionTaskContext) -> str:
        """Build display-safe document title."""
        return context.document_title or "Unknown Document"

    @staticmethod
    def _display_section_title(context: SectionTaskContext) -> str:
        """Build display-safe section title."""
        return context.section_title or "Untitled Section"
