"""Shared PDF mode vocabulary; no execution policy or I/O."""

from enum import Enum


class PdfExtractionMode(str, Enum):
    FAST = "fast"
    OCR = "ocr"
    AUTO = "auto"


class PdfExtractionProvenance(str, Enum):
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    APPLIED = "applied"
    REUSED = "reused"


class PdfExtractionFailureReason(str, Enum):
    MODE_UNAVAILABLE = "mode_unavailable"
    MODE_NOT_APPLICABLE = "mode_not_applicable"
    NO_TEXT = "no_text"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    QUALITY_FAILED = "quality_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    EXTRACTION_FAILED = "extraction_failed"


class PdfExtractionError(ValueError):
    """Typed preparation failure, mapped to HTTP only at the API boundary."""

    def __init__(self, reason: PdfExtractionFailureReason, detail: str):
        super().__init__(detail)
        self.reason = reason
