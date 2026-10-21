"""Internal OCR need metadata; constructing a decision never executes OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class PdfOcrDecision(str, Enum):
    SKIP = "skip"
    REQUIRED = "required"
    UNCERTAIN = "uncertain"


class PdfOcrPurpose(str, Enum):
    TEXT_RECOVERY = "text_recovery"
    LAYOUT_ANALYSIS = "layout_analysis"


class PdfOcrReason(str, Enum):
    NATIVE_TEXT_USABLE = "native_text_usable"
    BLANK_PAGE = "blank_page"
    NON_TEXT_PAGE = "non_text_page"
    TEXT_LAYER_MISSING = "text_layer_missing"
    TEXT_LAYER_UNRELIABLE = "text_layer_unreliable"
    UNCOVERED_IMAGE_TEXT = "uncovered_image_text"
    MIXED_CONTENT = "mixed_content"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    INSPECTION_FAILED = "inspection_failed"
    NATIVE_LAYOUT_SUFFICIENT = "native_layout_sufficient"
    LAYOUT_EVIDENCE_REQUIRED = "layout_evidence_required"
    SPARSE_NATIVE_TEXT = "sparse_native_text"
    HEADER_FOOTER_ONLY = "header_footer_only"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    VISUAL_TEXT_CONFIRMED = "visual_text_confirmed"


@dataclass(frozen=True)
class PdfPageOcrDecision:
    """A versioned need assessment for one source page and one OCR purpose."""

    source_sha256: str
    page_index: int
    purpose: PdfOcrPurpose
    decision: PdfOcrDecision
    reason_codes: tuple[PdfOcrReason, ...]
    policy_version: str
    evidence_schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.source_sha256
        ):
            raise ValueError("source_sha256 must be a PDF SHA-256 hex digest")
        if type(self.page_index) is not int or self.page_index < 0:
            raise ValueError("page_index must be a non-negative integer")
        if not isinstance(self.policy_version, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.policy_version
        ):
            raise ValueError("policy_version must be a bounded version identifier")
        if not isinstance(self.reason_codes, (tuple, list)) or not self.reason_codes:
            raise ValueError("reason_codes must be a non-empty sequence of known reasons")

        # Normalize a defensive copy so caller-owned lists cannot mutate evidence.
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())
        object.__setattr__(self, "purpose", PdfOcrPurpose(self.purpose))
        object.__setattr__(self, "decision", PdfOcrDecision(self.decision))
        reasons = tuple(dict.fromkeys(PdfOcrReason(code) for code in self.reason_codes))
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "purpose": self.purpose.value,
            "decision": self.decision.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "policy_version": self.policy_version,
            "evidence_schema_version": self.evidence_schema_version,
        }
