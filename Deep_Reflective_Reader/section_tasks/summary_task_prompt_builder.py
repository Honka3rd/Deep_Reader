from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext
from section_tasks.abstract_task_prompt_builder import AbstractTaskPromptBuilder
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon


class SummaryTaskPromptBuilder(AbstractTaskPromptBuilder):
    """Prompt builder dedicated to chapter summary task."""

    def __init__(self, common: SectionTaskPromptCommon):
        self.common = common

    def build(
        self,
        *,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
        language_code: LanguageCode | None = None,
    ) -> str:
        """Build one summary prompt from section context."""
        return (
            f"{self.common.build_header()}\n"
            "Task Type: summary\n"
            "Task:\n"
            "- Summarize only this section.\n"
            "- Keep output to 1-3 short paragraphs.\n"
            "- Do not include facts outside this section.\n"
            f"{self.common.build_language_instruction(language_code, document_profile)}\n"
            f"{self.common.build_topic_instruction(document_profile)}\n\n"
            f"{self.common.build_profile_block(document_profile)}\n"
            f"{self.common.build_context_block(context)}\n"
            "Section Content:\n"
            f"{context.section_content}\n"
        )
