from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext
from section_tasks.abstract_task_prompt_builder import AbstractTaskPromptBuilder
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon


class ChapterQuizTaskPromptBuilder(AbstractTaskPromptBuilder):
    """Prompt builder dedicated to chapter-level quiz task."""

    def __init__(self, common: SectionTaskPromptCommon):
        self.common = common

    def build(
        self,
        *,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
        language_code: LanguageCode | None = None,
    ) -> str:
        """Build chapter-quiz prompt from chapter context."""
        return (
            f"{self.common.build_header()}\n"
            "Task Type: chapter_quiz\n"
            "Task Scope:\n"
            "- This is a CHAPTER quiz.\n"
            "- Treat the provided section content as the full chapter context.\n"
            "- Ask questions that reflect chapter-level understanding.\n\n"
            "Task:\n"
            "- Create exactly 4 short-answer reading-comprehension questions.\n"
            "- Questions should cover key events, relationships, and turning points in this chapter.\n"
            "- Use only this chapter content.\n"
            "- Output must be strict JSON array only, without markdown fences.\n"
            "- Each item must include exactly these keys:\n"
            "  question_id, question_text, answer_text\n"
            "- question_id format: q1, q2, q3, q4.\n"
            "- answer_text must be concise and verifiable from chapter content.\n"
            f"{self.common.build_language_instruction(language_code, document_profile)}\n"
            f"{self.common.build_topic_instruction(document_profile)}\n\n"
            f"{self.common.build_profile_block(document_profile)}\n"
            f"{self.common.build_context_block(context)}\n"
            "Chapter Content:\n"
            f"{context.section_content}\n\n"
            "Output JSON example:\n"
            '[{"question_id":"q1","question_text":"...","answer_text":"..."}]\n'
        )
