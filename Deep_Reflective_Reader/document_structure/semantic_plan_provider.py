"""Request-local planner output allowance, shared by preflight and dispatch."""
import json
from document_structure.semantic_structure_plan import semantic_response_schema


class SemanticPlanProvider:
    def __init__(self, provider):
        self.provider = provider

    @property
    def supports_output_override(self):
        return callable(getattr(type(self.provider), "complete_text_with_output_budget", None))

    @property
    def effective_output_budget(self):
        if self.supports_output_override:
            return self.get_model_capabilities().max_output_tokens
        return getattr(self.provider, "effective_output_budget", None)

    def get_model_capabilities(self):
        return self.provider.get_model_capabilities()

    @property
    def request_schema_text(self):
        return json.dumps(semantic_response_schema(), separators=(",", ":")) if self.supports_output_override else ""

    def count_input_tokens(self, prompt):
        counter = getattr(self.provider, "count_input_tokens", None)
        return counter(prompt) if callable(counter) else None

    def complete_text(self, prompt):
        if self.supports_output_override:
            return self.provider.complete_text_with_output_budget(prompt, self.effective_output_budget,
                response_schema=semantic_response_schema())
        return self.provider.complete_text(prompt)
