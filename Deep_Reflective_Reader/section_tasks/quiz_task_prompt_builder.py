from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext
from section_tasks.section_task_prompt_builder import SectionTaskPromptBuilder
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon


class QuizTaskPromptBuilder(SectionTaskPromptBuilder):
    """Prompt builder dedicated to chapter quiz task."""

    def __init__(self, common: SectionTaskPromptCommon):
        self.common = common

    def build(
        self,
        *,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
    ) -> str:
        """Build one quiz prompt from section context."""
        return (
            f"{self.common.build_header()}\n"
            "Task Type: quiz\n"
            "Task:\n"
            "- Create exactly 4 short-answer reading-comprehension questions.\n"
            "- Use only this section.\n"
            "- Do not use information outside this section.\n"
            "- Output must be strict JSON array only, without markdown fences.\n"
            "- Each item must include exactly these keys:\n"
            "  question_id, question_text, answer_text\n"
            "- question_id format: q1, q2, q3, q4.\n"
            "- answer_text must be concise and verifiable from section content.\n"
            f"{self.common.build_language_instruction(document_profile)}\n"
            f"{self.common.build_topic_instruction(document_profile)}\n\n"
            f"{self.common.build_profile_block(document_profile)}\n"
            f"{self.common.build_context_block(context)}\n"
            "Section Content:\n"
            f"{context.section_content}\n\n"
            "Output JSON example:\n"
            '[{"question_id":"q1","question_text":"...","answer_text":"..."}]\n'
        )
