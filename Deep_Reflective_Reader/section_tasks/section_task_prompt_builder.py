from enum import StrEnum

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
        document_profile: object | None = None,
    ) -> str:
        """Build one task prompt from normalized task context and optional profile."""
        resolved_task_type = self._resolve_task_type(task_type)
        task_instruction = self._build_task_instruction(resolved_task_type)
        profile_block = self._build_profile_block(document_profile)
        language_instruction = self._build_language_instruction(document_profile)
        topic_instruction = self._build_topic_instruction(document_profile)

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
            f"- Document Title: {context.document_title}\n"
            f"- Container Title: {context.container_title}\n"
            f"- Section ID: {context.section_id}\n"
            f"- Section Title: {context.section_title}\n"
            f"- Section Index: {context.section_index}\n"
            f"- Context Valid: {context.valid}\n"
            f"- Context Reason: {context.reason}\n\n"
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

    def _build_profile_block(self, document_profile: object | None) -> str:
        """Return compact profile background block with compatibility extraction."""
        topic = self._extract_profile_value(
            document_profile,
            "topic",
            "document_topic",
        )
        language = self._extract_profile_value(
            document_profile,
            "document_language",
            "language",
            "language_code",
        )
        summary = self._extract_profile_value(
            document_profile,
            "summary",
            "document_summary",
            "profile_summary",
        )

        if topic is None and language is None and summary is None:
            return "Document Profile: None"

        return (
            "Document Profile (background only):\n"
            f"- Topic: {topic or 'unknown'}\n"
            f"- Language: {language or 'unknown'}\n"
            f"- Summary: {summary or 'none'}"
        )

    def _build_language_instruction(self, document_profile: object | None) -> str:
        """Return output language instruction based on profile language signal."""
        language = self._extract_profile_value(
            document_profile,
            "document_language",
            "language",
            "language_code",
        )
        normalized = (language or "").strip().lower()
        if normalized.startswith("zh"):
            return "Output Language: Chinese"
        if normalized.startswith("en"):
            return "Output Language: English"
        return "Output Language: Follow the section content language"

    def _build_topic_instruction(self, document_profile: object | None) -> str:
        """Return light topic-aware instruction without overriding section evidence."""
        topic = self._extract_profile_value(
            document_profile,
            "topic",
            "document_topic",
        )
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
    def _extract_profile_value(document_profile: object | None, *keys: str) -> str | None:
        """Extract one profile field with basic dict/object compatibility."""
        if document_profile is None:
            return None
        for key in keys:
            value: object | None = None
            if isinstance(document_profile, dict):
                value = document_profile.get(key)
            else:
                value = getattr(document_profile, key, None)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                return text_value
        return None
