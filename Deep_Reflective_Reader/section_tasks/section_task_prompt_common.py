from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext


class SectionTaskPromptCommon:
    """Shared prompt helper for section-based task prompt builders."""

    def build_header(self) -> str:
        """Return task-agnostic instruction header."""
        return (
            "You are working on one section-based reading task.\n"
            "Use section content as the primary evidence.\n"
            "Document profile is optional background only.\n"
        )

    def build_profile_block(self, document_profile: DocumentProfile | None) -> str:
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

    def build_language_instruction(self, document_profile: DocumentProfile | None) -> str:
        """Return output language instruction based on profile language signal."""
        language = self._extract_profile_language(document_profile)
        normalized = (language or "").strip().lower()
        if normalized.startswith("zh"):
            return "Output Language: Chinese"
        if normalized.startswith("en"):
            return "Output Language: English"
        return "Output Language: Follow the section content language"

    def build_topic_instruction(self, document_profile: DocumentProfile | None) -> str:
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

    def build_context_block(self, context: SectionTaskContext) -> str:
        """Build section context block shared by different task prompts."""
        container_title_line = self._build_container_title_line(context)
        return (
            "Section Context:\n"
            f"- Document Title: {self._display_document_title(context)}\n"
            f"{container_title_line}"
            f"- Section ID: {context.section_id}\n"
            f"- Section Title: {self._display_section_title(context)}\n"
            f"- Section Index: {context.section_index}\n"
        )

    @staticmethod
    def _extract_profile_topic(document_profile: DocumentProfile | None) -> str | None:
        if document_profile is None:
            return None
        text_value = str(document_profile.topic).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _extract_profile_language(document_profile: DocumentProfile | None) -> str | None:
        if document_profile is None:
            return None
        text_value = str(document_profile.document_language).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _extract_profile_summary(document_profile: DocumentProfile | None) -> str | None:
        if document_profile is None:
            return None
        text_value = str(document_profile.summary).strip()
        if text_value:
            return text_value
        return None

    @staticmethod
    def _build_container_title_line(context: SectionTaskContext) -> str:
        if context.container_title is None:
            return ""
        return f"- Container Title: {context.container_title}\n"

    @staticmethod
    def _display_document_title(context: SectionTaskContext) -> str:
        return context.document_title or "Unknown Document"

    @staticmethod
    def _display_section_title(context: SectionTaskContext) -> str:
        return context.section_title or "Untitled Section"
