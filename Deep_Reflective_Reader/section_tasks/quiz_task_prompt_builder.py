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
            "- Create 3 to 5 short-answer reading-comprehension questions.\n"
            "- Provide concise standard answers.\n"
            "- Use only this section.\n"
            "- Output in plain text with Q1/A1, Q2/A2 format.\n"
            f"{self.common.build_language_instruction(document_profile)}\n"
            f"{self.common.build_topic_instruction(document_profile)}\n\n"
            f"{self.common.build_profile_block(document_profile)}\n"
            f"{self.common.build_context_block(context)}\n"
            "Section Content:\n"
            f"{context.section_content}\n"
        )
