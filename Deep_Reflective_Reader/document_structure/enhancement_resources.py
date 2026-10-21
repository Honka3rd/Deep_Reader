"""Request-local assembly limits, distinct from model context capacity."""
import logging
from dataclasses import dataclass

from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
from document_structure.source_mapped_text import MAX_SOURCE_CHARS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancementResourceLimits:
    max_source_chars: int = MAX_SOURCE_CHARS
    max_source_bytes: int = 8_000_000
    max_line_breaks: int = 100_000
    max_prompt_bytes: int = 16_000_000

    def __post_init__(self):
        for value in (self.max_source_chars, self.max_source_bytes,
                      self.max_line_breaks, self.max_prompt_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("invalid enhancement resource limit")
        if self.max_source_chars > MAX_SOURCE_CHARS:
            raise ValueError("resource limit exceeds source mapper support")

    def check_source(self, source):
        self._check("source_chars", len(source), self.max_source_chars)
        # Count before allocating line lists, source maps or encoded copies.
        breaks = sum(source.count(char) for char in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
        self._check("line_breaks", breaks, self.max_line_breaks)
        size = len(source.encode("utf-8"))
        self._check("source_bytes", size, self.max_source_bytes)
        return size

    def check_prompt(self, prompt):
        self._check("prompt_chars", len(prompt), self.max_prompt_bytes)
        size = len(prompt.encode("utf-8"))
        self._check("prompt_bytes", size, self.max_prompt_bytes)
        return size

    @staticmethod
    def _check(kind, size, limit):
        if size > limit:
            LOGGER.warning("enhancement_resource_rejected reason=local_resource_limit kind=%s size=%d limit=%d",
                           kind, size, limit)
            raise EnhancementPromptBudgetError("local_resource_limit")
