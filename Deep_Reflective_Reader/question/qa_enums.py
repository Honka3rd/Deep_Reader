from enum import StrEnum


class AnswerLevel(StrEnum):
    """Strictness level derived from retrieval relevance."""
    STRICT = "strict"
    CAUTIOUS = "cautious"
    REJECT = "reject"


class QuestionScope(StrEnum):
    """Question scope classification for context orchestration."""
    LOCAL = "local"
    GLOBAL = "global"


class LocalSignalQuality(StrEnum):
    """Quality tier for local-reference signals."""
    STRONG = "strong"
    WEAK = "weak"


class PromptMode(StrEnum):
    """Prompt-level context mode passed to prompt assembler."""
    LOCAL_READING = "local_reading_mode"
    RETRIEVAL = "retrieval_mode"
    FULL_TEXT = "full_text_mode"


class ContextMode(StrEnum):
    """Context construction mode produced by orchestrator."""
    LOCAL_WINDOW = "local_window_mode"
    RETRIEVAL = "retrieval_mode"
    FULL_TEXT = "full_text_mode"
