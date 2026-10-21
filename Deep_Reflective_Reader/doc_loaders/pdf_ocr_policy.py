"""Pure OCR need assessment. No renderer, detector, OCR or persistence is invoked."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

from .pdf_ocr_decision import PdfOcrDecision, PdfOcrPurpose, PdfOcrReason, PdfPageOcrDecision
from .pdf_page_inspection import PdfNativeTextQuality, PdfPageInspection


def _ratio(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class PdfOcrPolicyConfig:
    min_native_chars: int = 40
    min_printable_ratio: float = 0.98
    max_replacement_ratio: float = 0.01
    max_control_ratio: float = 0.0
    max_repeat_ratio: float = 0.35
    max_duplicate_line_ratio: float = 0.5
    body_margin_ratio: float = 0.12
    min_layout_span_ratio: float = 0.15
    probe_attempt_limit: int = 1
    probe_dpi: int = 96
    probe_max_pixels: int = 1_000_000
    probe_timeout_seconds: int = 5

    def __post_init__(self) -> None:
        for name in (
            "min_printable_ratio", "max_replacement_ratio", "max_control_ratio",
            "max_repeat_ratio", "max_duplicate_line_ratio", "min_layout_span_ratio",
            "body_margin_ratio",
        ):
            if not _ratio(getattr(self, name)):
                raise ValueError(f"{name} must be a finite ratio")
        if self.body_margin_ratio >= 0.5:
            raise ValueError("body_margin_ratio must leave an interior")
        for name, low, high in (
            ("min_native_chars", 1, 100_000), ("probe_attempt_limit", 0, 2),
            ("probe_dpi", 36, 120), ("probe_max_pixels", 256, 2_000_000),
            ("probe_timeout_seconds", 1, 10),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")

    @property
    def policy_version(self) -> str:
        fingerprint = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]
        return f"pdf-ocr-policy-v2-{fingerprint}"


@dataclass(frozen=True)
class PdfPageVisualCheck:
    """Optional output of a future bounded detector, never an inferred image label."""

    source_sha256: str
    page_index: int
    detector_version: str
    contains_text: bool | None
    covers_full_page: bool
    native_text_complete: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a PDF digest")
        if type(self.page_index) is not int or self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not isinstance(self.detector_version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.detector_version):
            raise ValueError("detector_version must be a bounded identifier")
        for name in ("contains_text", "native_text_complete"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be bool or None")
        if type(self.covers_full_page) is not bool:
            raise ValueError("covers_full_page must be bool")
        if self.native_text_complete is not None and self.contains_text is not True:
            raise ValueError("native coverage requires positive text evidence")
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())


@dataclass(frozen=True)
class PdfOcrProbePlan:
    status: str
    dpi: int | None = None
    max_pixels: int | None = None
    timeout_seconds: int | None = None


class PdfOcrPolicy:
    def __init__(self, config: PdfOcrPolicyConfig | None = None):
        self.config = config or PdfOcrPolicyConfig()

    def assess(
        self, page: PdfPageInspection, purpose: PdfOcrPurpose,
        *, visual_check: PdfPageVisualCheck | None = None,
    ) -> PdfPageOcrDecision:
        purpose = PdfOcrPurpose(purpose)

        def result(state, *reasons):
            return PdfPageOcrDecision(
                source_sha256=page.source_sha256, page_index=page.page_index,
                purpose=purpose, decision=state, reason_codes=reasons,
                policy_version=self.config.policy_version,
            )

        skip, required, uncertain = PdfOcrDecision
        reason = PdfOcrReason
        if visual_check is not None and (
            visual_check.source_sha256 != page.source_sha256.lower()
            or visual_check.page_index != page.page_index
        ):
            raise ValueError("visual evidence does not match source/page")
        q = page.text_quality
        if page.evidence_schema_version != 1 or not self._valid_quality(q, page.native_text):
            return result(uncertain, reason.INSPECTION_FAILED)
        if {"native_text_extraction_failed", "content_stream_inspection_failed", "resources_unavailable"} & set(page.flags):
            return result(uncertain, reason.INSPECTION_FAILED)

        quality_ok = self._quality_ok(q)
        # Partial/unknown visual checks never authorize whole-page skip or OCR.
        if visual_check is not None:
            if not visual_check.covers_full_page or visual_check.contains_text is None:
                return result(uncertain, reason.EVIDENCE_INCOMPLETE)
            if not visual_check.contains_text:
                if q.non_whitespace_count:
                    return result(uncertain, reason.CONFLICTING_EVIDENCE)
                return result(skip, reason.NON_TEXT_PAGE)
            if visual_check.native_text_complete is True and (not q.non_whitespace_count or not quality_ok):
                return result(uncertain, reason.CONFLICTING_EVIDENCE)
            if not q.non_whitespace_count or not quality_ok or visual_check.native_text_complete is False:
                why = (reason.TEXT_LAYER_MISSING if not q.non_whitespace_count
                       else reason.TEXT_LAYER_UNRELIABLE if not quality_ok
                       else reason.UNCOVERED_IMAGE_TEXT)
                if purpose is PdfOcrPurpose.LAYOUT_ANALYSIS:
                    return result(required, reason.LAYOUT_EVIDENCE_REQUIRED, reason.VISUAL_TEXT_CONFIRMED)
                return result(required, why, reason.VISUAL_TEXT_CONFIRMED)
            if visual_check.native_text_complete is True:
                if purpose is PdfOcrPurpose.TEXT_RECOVERY:
                    return result(skip, reason.NATIVE_TEXT_USABLE)
                if self._layout_sufficient(page):
                    return result(skip, reason.NATIVE_LAYOUT_SUFFICIENT)
            return result(uncertain, reason.EVIDENCE_INCOMPLETE)

        if not self._geometry_known(page) or set(page.flags) - {"inline_image_present"}:
            return result(uncertain, reason.EVIDENCE_INCOMPLETE)
        no_images = page.image_draw_count == 0 and page.image_coverage_ratio == 0
        if not q.non_whitespace_count:
            if no_images and page.has_vector_paint is False:
                return result(skip, reason.BLANK_PAGE)
            return result(uncertain, reason.TEXT_LAYER_MISSING, reason.EVIDENCE_INCOMPLETE)
        if not quality_ok:
            return result(uncertain, reason.TEXT_LAYER_UNRELIABLE)
        if not no_images or (page.has_vector_paint is not False and not page.has_only_page_background):
            return result(uncertain, reason.MIXED_CONTENT, reason.EVIDENCE_INCOMPLETE)
        if q.non_whitespace_count < self.config.min_native_chars:
            return result(uncertain, reason.SPARSE_NATIVE_TEXT)
        if page.has_font_resources is not True or not self._positions_known(page):
            return result(uncertain, reason.EVIDENCE_INCOMPLETE)
        left, bottom, right, top = page.crop_box
        margin = self.config.body_margin_ratio
        if not any(margin < (y - bottom) / (top - bottom) < 1 - margin for _, y in page.text_origins):
            return result(uncertain, reason.HEADER_FOOTER_ONLY)
        if purpose is PdfOcrPurpose.TEXT_RECOVERY:
            return result(skip, reason.NATIVE_TEXT_USABLE)
        if self._layout_sufficient(page):
            return result(skip, reason.NATIVE_LAYOUT_SUFFICIENT)
        return result(uncertain, reason.EVIDENCE_INCOMPLETE)

    def plan_probe(
        self, decision: PdfPageOcrDecision, *, attempts_used: int,
        ocr_enabled: bool, detector_available: bool,
    ) -> PdfOcrProbePlan:
        """Caller shares attempt count across both purposes; this returns limits only."""
        if type(attempts_used) is not int or attempts_used < 0:
            raise ValueError("attempts_used must be non-negative")
        if type(ocr_enabled) is not bool or type(detector_available) is not bool:
            raise ValueError("execution gates must be bool")
        if decision.policy_version != self.config.policy_version:
            raise ValueError("decision belongs to another policy configuration")
        if decision.decision is not PdfOcrDecision.UNCERTAIN:
            return PdfOcrProbePlan("not_needed")
        if not ocr_enabled:
            return PdfOcrProbePlan("disabled")
        if not detector_available:
            return PdfOcrProbePlan("unavailable")
        if attempts_used >= self.config.probe_attempt_limit:
            return PdfOcrProbePlan("budget_exhausted")
        return PdfOcrProbePlan("probe", self.config.probe_dpi,
                               self.config.probe_max_pixels, self.config.probe_timeout_seconds)

    @staticmethod
    def _valid_quality(q: PdfNativeTextQuality | None, text: str | None) -> bool:
        if q is None or text is None or q.character_count != len(text):
            return False
        if type(q.non_whitespace_count) is not int or not 0 <= q.non_whitespace_count <= q.character_count:
            return False
        return all(_ratio(getattr(q, name)) for name in (
            "printable_ratio", "replacement_ratio", "control_ratio", "longest_repeat_ratio", "duplicate_line_ratio",
        ))

    def _quality_ok(self, q: PdfNativeTextQuality) -> bool:
        c = self.config
        return (q.printable_ratio >= c.min_printable_ratio
                and q.replacement_ratio <= c.max_replacement_ratio
                and q.control_ratio <= c.max_control_ratio
                and (q.non_whitespace_count < c.min_native_chars or q.longest_repeat_ratio <= c.max_repeat_ratio)
                and q.duplicate_line_ratio <= c.max_duplicate_line_ratio)

    @staticmethod
    def _geometry_known(page: PdfPageInspection) -> bool:
        box = page.crop_box
        return (box is not None and len(box) == 4 and all(math.isfinite(v) for v in box)
                and box[2] > box[0] and box[3] > box[1]
                and type(page.image_draw_count) is int and page.image_draw_count >= 0
                and _ratio(page.image_coverage_ratio)
                and page.rotation_degrees in (0, 90, 180, 270)
                and type(page.has_vector_paint) is bool
                and not (page.image_draw_count == 0 and page.image_coverage_ratio != 0))

    def _positions_known(self, page: PdfPageInspection) -> bool:
        if not self._geometry_known(page) or not page.text_positions_complete or not page.text_origins:
            return False
        l, b, r, t = page.crop_box
        return all(len(point) == 2 and all(math.isfinite(v) for v in point)
                   and l <= point[0] <= r and b <= point[1] <= t for point in page.text_origins)

    def _layout_sufficient(self, page: PdfPageInspection) -> bool:
        if (not self._positions_known(page) or page.rotation_degrees != 0
                or page.image_draw_count != 0 or (page.has_vector_paint is not False and not page.has_only_page_background)
                or page.has_font_resources is not True or page.flags):
            return False
        ys = [y for _, y in page.text_origins]
        return (max(ys) - min(ys)) / (page.crop_box[3] - page.crop_box[1]) >= self.config.min_layout_span_ratio
