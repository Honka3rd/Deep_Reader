"""Shared measured model capacity for learning translation and hints; no IO writes."""
from section_tasks.learning_generation_policy import BatchPlan
from llm.llm_model_capabilities import LLMModelCapabilities
from shared.learning_source import token

FRAMING_TOKENS = 256  # existing transport reserve, not a source/output estimate


class LearningGenerationCapacity:
    def __init__(self, provider, error_type, prefix):
        self.provider, self.error_type, self.prefix = provider, error_type, prefix

    def error(self, suffix):
        return self.error_type(self.prefix + suffix)

    def capabilities(self):
        try:
            caps = self.provider.get_model_capabilities()
            if type(caps) is not LLMModelCapabilities:
                raise ValueError('capabilities')
            token(caps.model_name)
            caps.input_token_limit(caps.max_output_tokens)
            override = callable(getattr(self.provider, 'complete_text_with_output_budget', None))
            if not override and getattr(self.provider, 'effective_output_budget', None) != caps.max_output_tokens:
                raise ValueError('cannot dispatch model output ceiling')
            return caps
        except Exception:
            raise self.error('capacity_unavailable') from None

    def fits(self, prompt, caps):
        try:
            count = self.provider.count_input_tokens(prompt)
            if type(count) is not int or count <= 0:
                raise ValueError('tokenizer unavailable')
            return count + FRAMING_TOKENS <= caps.input_token_limit(caps.max_output_tokens)
        except Exception:
            raise self.error('capacity_unavailable') from None

    def plan(self, sentences, caps, prompt_for_group):
        groups, pending = [], [tuple(s.sentence_id for s in sentences.sentences)]
        # Complete preflight, <= 2*N-1 measured candidates; never clip a sentence.
        while pending:
            group = pending.pop()
            if self.fits(prompt_for_group(group), caps):
                groups.append(group)
            elif len(group) == 1:
                raise self.error('sentence_capacity_exceeded')
            else:
                mid = len(group) // 2
                pending.extend((group[mid:], group[:mid]))
        plan = BatchPlan(tuple(groups))
        plan.validate_against(sentences)
        return plan
