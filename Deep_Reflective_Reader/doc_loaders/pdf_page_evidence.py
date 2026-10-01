"""Page-level PDF layout evidence used by deterministic structure analysis."""

from __future__ import annotations

import csv
import io
import re
import statistics
from dataclasses import dataclass, field, replace


OCR_TOKEN_NORMALIZATION_VERSION = 1


@dataclass(frozen=True)
class PdfPageTextBoundary:
    """Exact raw-text span belonging to one PDF page in canonical page order."""

    page_index: int
    char_start: int
    char_end: int
    text: str
    page_label: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class PdfPageBoundaryEvidenceResult:
    """Compact source evidence for resolving manual page anchors to raw text."""

    doc_name: str
    source_file_name: str
    source_sha256: str
    page_count: int
    boundaries: list[PdfPageTextBoundary]
    evidence_schema_version: int = 1


@dataclass(frozen=True)
class PdfOcrWordEvidence:
    """One OCR word/token with source page coordinates."""

    page_index: int
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    normalized_text: str = ""
    normalization_version: int = OCR_TOKEN_NORMALIZATION_VERSION
    region_id: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    block_num: int | None = None
    par_num: int | None = None
    line_num: int | None = None
    word_num: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "normalization_version": self.normalization_version,
            "confidence": self.confidence,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "region_id": self.region_id,
            "quality_flags": list(self.quality_flags),
            "block_num": self.block_num,
            "par_num": self.par_num,
            "line_num": self.line_num,
            "word_num": self.word_num,
        }


@dataclass(frozen=True)
class PdfOcrRegionEvidence:
    """OCR region evidence with page-local geometry and reading order."""

    region_id: str
    page_index: int
    region_type: str
    left: int
    top: int
    right: int
    bottom: int
    writing_mode: str
    reading_order: str
    raw_ocr: str
    normalized_text: str
    confidence: float
    token_count: int
    rotation_degrees: int | None = None
    quality_flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "region_type": self.region_type,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "writing_mode": self.writing_mode,
            "reading_order": self.reading_order,
            "raw_ocr": self.raw_ocr,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "rotation_degrees": self.rotation_degrees,
            "quality_flags": list(self.quality_flags),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class PdfPageLayoutEvidence:
    page_index: int
    width: int | None
    height: int | None
    native_text_chars: int
    image_count: int
    source_file_name: str | None = None
    source_sha256: str | None = None
    evidence_schema_version: int = 1
    orientation: str = "unknown"
    rotation_degrees: int | None = None
    orientation_hypotheses: list[dict[str, object]] = field(default_factory=list)
    writing_mode: str = "unknown"
    reading_order: str = "unknown"
    writing_mode_confidence: float = 0.0
    reading_order_confidence: float = 0.0
    ocr_confidence: float = 0.0
    ocr_text: str = ""
    word_count: int = 0
    region_count: int = 0
    column_count: int = 0
    line_count: int = 0
    analysis_stage: str = "inventory"
    analysis_cost_tier: str = "cheap_inventory"
    ocr_engine: str | None = None
    ocr_language: str | None = None
    render_dpi: int | None = None
    ocr_pass_count: int = 0
    cache_key: str | None = None
    cache_hit: bool | None = None
    failure_reason: str | None = None
    high_cost_analysis_permitted: bool = False
    evidence: list[str] = field(default_factory=list)
    ocr_words: list[PdfOcrWordEvidence] = field(default_factory=list)
    ocr_regions: list[PdfOcrRegionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "native_text_chars": self.native_text_chars,
            "image_count": self.image_count,
            "source_file_name": self.source_file_name,
            "source_sha256": self.source_sha256,
            "evidence_schema_version": self.evidence_schema_version,
            "orientation": self.orientation,
            "rotation_degrees": self.rotation_degrees,
            "orientation_hypotheses": [dict(item) for item in self.orientation_hypotheses],
            "writing_mode": self.writing_mode,
            "reading_order": self.reading_order,
            "writing_mode_confidence": self.writing_mode_confidence,
            "reading_order_confidence": self.reading_order_confidence,
            "ocr_confidence": self.ocr_confidence,
            "ocr_text": self.ocr_text,
            "word_count": self.word_count,
            "region_count": self.region_count,
            "column_count": self.column_count,
            "line_count": self.line_count,
            "analysis_stage": self.analysis_stage,
            "analysis_cost_tier": self.analysis_cost_tier,
            "ocr_engine": self.ocr_engine,
            "ocr_language": self.ocr_language,
            "render_dpi": self.render_dpi,
            "ocr_pass_count": self.ocr_pass_count,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "failure_reason": self.failure_reason,
            "high_cost_analysis_permitted": self.high_cost_analysis_permitted,
            "evidence": list(self.evidence),
            "ocr_words": [word.to_dict() for word in self.ocr_words],
            "ocr_regions": [region.to_dict() for region in self.ocr_regions],
        }


def analyze_ocr_tsv(
    *,
    page_index: int,
    width: int | None,
    height: int | None,
    native_text_chars: int,
    image_count: int,
    tsv_text: str,
    source_file_name: str | None = None,
    source_sha256: str | None = None,
    evidence_schema_version: int = 1,
    analysis_cost_tier: str = "coordinate_ocr",
    ocr_engine: str | None = "tesseract",
    ocr_language: str | None = None,
    render_dpi: int | None = None,
    ocr_pass_count: int = 1,
    cache_key: str | None = None,
    cache_hit: bool | None = False,
    failure_reason: str | None = None,
    high_cost_analysis_permitted: bool = False,
) -> PdfPageLayoutEvidence:
    """Convert Tesseract TSV into compact layout hypotheses."""
    orientation, rotation_degrees, orientation_hypotheses, orientation_evidence = (
        _orientation_hypotheses(width, height)
    )
    rows = list(csv.DictReader(io.StringIO(tsv_text), delimiter="\t"))
    words: list[PdfOcrWordEvidence] = []
    line_boxes: list[dict[str, int]] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf", "-1"))
            left = int(row.get("left", "0"))
            top = int(row.get("top", "0"))
            word_width = int(row.get("width", "0"))
            word_height = int(row.get("height", "0"))
        except (TypeError, ValueError):
            continue
        if row.get("level") == "4" and word_width > 0 and word_height > 0:
            line_boxes.append(
                {"left": left, "top": top, "width": word_width, "height": word_height}
            )
        if not text or confidence < 0 or word_width <= 0 or word_height <= 0:
            continue
        words.append(
            PdfOcrWordEvidence(
                page_index=page_index,
                text=text,
                normalized_text=_normalize_ocr_token(text),
                confidence=confidence,
                left=left,
                top=top,
                width=word_width,
                height=word_height,
                quality_flags=_ocr_word_quality_flags(text=text, confidence=confidence),
                block_num=_optional_int(row.get("block_num")),
                par_num=_optional_int(row.get("par_num")),
                line_num=_optional_int(row.get("line_num")),
                word_num=_optional_int(row.get("word_num")),
            )
        )

    if not words:
        return PdfPageLayoutEvidence(
            page_index=page_index,
            width=width,
            height=height,
            native_text_chars=native_text_chars,
            image_count=image_count,
            source_file_name=source_file_name,
            source_sha256=source_sha256,
            evidence_schema_version=evidence_schema_version,
            orientation=orientation,
            rotation_degrees=rotation_degrees,
            orientation_hypotheses=orientation_hypotheses,
            analysis_stage="coordinate_ocr",
            analysis_cost_tier=analysis_cost_tier,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            render_dpi=render_dpi,
            ocr_pass_count=ocr_pass_count,
            cache_key=cache_key,
            cache_hit=cache_hit,
            failure_reason=failure_reason,
            high_cost_analysis_permitted=high_cost_analysis_permitted,
            evidence=["no_ocr_words", *orientation_evidence],
        )

    geometry = line_boxes or words
    median_width = statistics.median(_geometry_width(item) for item in geometry)
    median_height = statistics.median(_geometry_height(item) for item in geometry)
    x_centers = sorted(
        {round(_geometry_left(item) + _geometry_width(item) / 2) for item in geometry}
    )
    line_ids = (
        {str(word.line_num) for word in words if word.line_num is not None}
        if not line_boxes
        else {str(index) for index in range(len(line_boxes))}
    )
    vertical_shape = median_height > median_width * 1.35
    column_count = _cluster_count(x_centers, max(24, int(median_width * 1.5)))
    region_count = max(1, column_count)
    words = _assign_word_region_ids(
        page_index=page_index,
        words=words,
        region_count=region_count,
        x_centers=x_centers,
        page_height=height,
    )
    writing_mode = "vertical" if vertical_shape and column_count >= 2 else "horizontal"
    reading_order = "right_to_left" if writing_mode == "vertical" else "left_to_right"
    ocr_regions = _build_ocr_regions(
        page_index=page_index,
        words=words,
        page_height=height,
        writing_mode=writing_mode,
        reading_order=reading_order,
        rotation_degrees=rotation_degrees,
    )
    writing_mode_confidence = _writing_mode_confidence(
        median_width=median_width,
        median_height=median_height,
        column_count=column_count,
        writing_mode=writing_mode,
    )
    reading_order_confidence = _reading_order_confidence(
        writing_mode=writing_mode,
        column_count=column_count,
        line_count=len(line_ids),
    )
    text = "\n".join(word.text for word in words)
    confidence = statistics.mean(word.confidence for word in words) / 100.0
    evidence = [
        "coordinate_ocr",
        "vertical_glyph_geometry" if vertical_shape else "horizontal_glyph_geometry",
        *orientation_evidence,
    ]
    if column_count >= 2:
        evidence.append("multiple_text_columns")
    if 1.15 <= (median_height / max(median_width, 1)) <= 1.45:
        evidence.append("ambiguous_writing_mode")
    return PdfPageLayoutEvidence(
        page_index=page_index,
        width=width,
        height=height,
        native_text_chars=native_text_chars,
        image_count=image_count,
        source_file_name=source_file_name,
        source_sha256=source_sha256,
        evidence_schema_version=evidence_schema_version,
        orientation=orientation,
        rotation_degrees=rotation_degrees,
        orientation_hypotheses=orientation_hypotheses,
        writing_mode=writing_mode,
        reading_order=reading_order,
        writing_mode_confidence=writing_mode_confidence,
        reading_order_confidence=reading_order_confidence,
        ocr_confidence=round(confidence, 4),
        ocr_text=text,
        word_count=len(words),
        region_count=len(ocr_regions),
        column_count=column_count,
        line_count=len(line_ids),
        analysis_stage="coordinate_ocr",
        analysis_cost_tier=analysis_cost_tier,
        ocr_engine=ocr_engine,
        ocr_language=ocr_language,
        render_dpi=render_dpi,
        ocr_pass_count=ocr_pass_count,
        cache_key=cache_key,
        cache_hit=cache_hit,
        failure_reason=failure_reason,
        high_cost_analysis_permitted=high_cost_analysis_permitted,
        evidence=evidence,
        ocr_words=words,
        ocr_regions=ocr_regions,
    )


def _orientation_hypotheses(
    width: int | None,
    height: int | None,
) -> tuple[str, int | None, list[dict[str, object]], list[str]]:
    if not width or not height:
        return (
            "unknown",
            None,
            [
                {
                    "orientation": "unknown",
                    "rotation_degrees": None,
                    "confidence": 0.0,
                    "evidence": ["missing_page_dimensions"],
                }
            ],
            ["missing_page_dimensions"],
        )

    dimension_delta = abs(height - width) / max(height, width)
    if dimension_delta <= 0.08:
        return (
            "unknown",
            None,
            [
                {
                    "orientation": "portrait",
                    "rotation_degrees": 0,
                    "confidence": 0.5,
                    "evidence": ["near_square_page_dimensions"],
                },
                {
                    "orientation": "landscape",
                    "rotation_degrees": 90,
                    "confidence": 0.5,
                    "evidence": ["near_square_page_dimensions"],
                },
            ],
            ["ambiguous_orientation", "near_square_page_dimensions"],
        )

    if height > width:
        return (
            "portrait",
            0,
            [
                {
                    "orientation": "portrait",
                    "rotation_degrees": 0,
                    "confidence": round(min(0.98, 0.55 + dimension_delta), 4),
                    "evidence": ["page_dimensions"],
                },
                {
                    "orientation": "landscape",
                    "rotation_degrees": 90,
                    "confidence": round(max(0.05, 1.0 - dimension_delta), 4),
                    "evidence": ["dimension_alternate"],
                },
            ],
            ["page_dimensions_orientation"],
        )

    return (
        "landscape",
        90,
        [
            {
                "orientation": "landscape",
                "rotation_degrees": 90,
                "confidence": round(min(0.98, 0.55 + dimension_delta), 4),
                "evidence": ["page_dimensions"],
            },
            {
                "orientation": "portrait",
                "rotation_degrees": 0,
                "confidence": round(max(0.05, 1.0 - dimension_delta), 4),
                "evidence": ["dimension_alternate"],
            },
        ],
        ["page_dimensions_orientation"],
    )


def _writing_mode_confidence(
    *,
    median_width: float,
    median_height: float,
    column_count: int,
    writing_mode: str,
) -> float:
    glyph_ratio = median_height / max(median_width, 1)
    if writing_mode == "vertical":
        confidence = 0.5 + min(0.25, (glyph_ratio - 1.35) * 0.2)
        confidence += 0.2 if column_count >= 2 else 0.0
        return round(min(0.95, max(0.0, confidence)), 4)
    confidence = 0.55 + min(0.25, max(0.0, 1.35 - glyph_ratio) * 0.2)
    if column_count <= 1:
        confidence += 0.1
    return round(min(0.95, max(0.0, confidence)), 4)


def _reading_order_confidence(
    *,
    writing_mode: str,
    column_count: int,
    line_count: int,
) -> float:
    if writing_mode == "vertical":
        confidence = 0.55 + (0.2 if column_count >= 2 else 0.0)
    else:
        confidence = 0.6 + (0.15 if line_count >= 2 else 0.0)
    return round(min(0.95, confidence), 4)


def _assign_word_region_ids(
    *,
    page_index: int,
    words: list[PdfOcrWordEvidence],
    region_count: int,
    x_centers: list[int],
    page_height: int | None,
) -> list[PdfOcrWordEvidence]:
    if not words:
        return []
    if region_count <= 1 or not x_centers:
        return [
            replace(
                word,
                region_id=_word_region_id(
                    page_index=page_index,
                    word=word,
                    region_index=0,
                    page_height=page_height,
                ),
            )
            for word in words
        ]

    centers = _region_centers(x_centers, region_count)
    return [
        replace(
            word,
            region_id=_word_region_id(
                page_index=page_index,
                word=word,
                region_index=_nearest_region_index(
                    _geometry_left(word) + _geometry_width(word) / 2,
                    centers,
                ),
                page_height=page_height,
            ),
        )
        for word in words
    ]


def _word_region_id(
    *,
    page_index: int,
    word: PdfOcrWordEvidence,
    region_index: int,
    page_height: int | None,
) -> str:
    region_type = _word_region_type(word=word, page_height=page_height)
    if region_type in {"header", "footer"}:
        return f"page:{page_index}:{region_type}"
    return f"page:{page_index}:region:{region_index}"


def _build_ocr_regions(
    *,
    page_index: int,
    words: list[PdfOcrWordEvidence],
    page_height: int | None,
    writing_mode: str,
    reading_order: str,
    rotation_degrees: int | None,
) -> list[PdfOcrRegionEvidence]:
    grouped: dict[str, list[PdfOcrWordEvidence]] = {}
    for word in words:
        region_id = word.region_id or f"page:{page_index}:region:0"
        grouped.setdefault(region_id, []).append(word)

    regions: list[PdfOcrRegionEvidence] = []
    for region_id, region_words in grouped.items():
        ordered_words = _sort_region_words(region_words, writing_mode=writing_mode)
        left = min(word.left for word in region_words)
        top = min(word.top for word in region_words)
        right = max(word.left + word.width for word in region_words)
        bottom = max(word.top + word.height for word in region_words)
        region_type = _region_type(
            region_id=region_id,
            top=top,
            bottom=bottom,
            page_height=page_height,
        )
        raw_ocr = _join_region_text(ordered_words, writing_mode=writing_mode, raw=True)
        normalized_text = _join_region_text(
            ordered_words,
            writing_mode=writing_mode,
            raw=False,
        )
        quality_flags = sorted(
            {flag for word in region_words for flag in word.quality_flags}
        )
        evidence = [
            "region_from_ocr_words",
            f"{writing_mode}_region",
            f"region_type:{region_type}",
        ]
        if reading_order == "right_to_left":
            evidence.append("right_to_left_region_order")
        if rotation_degrees is not None:
            evidence.append(f"rotation:{rotation_degrees}")
        regions.append(
            PdfOcrRegionEvidence(
                region_id=region_id,
                page_index=page_index,
                region_type=region_type,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                writing_mode=writing_mode,
                reading_order=reading_order,
                raw_ocr=raw_ocr,
                normalized_text=normalized_text,
                confidence=round(
                    statistics.mean(word.confidence for word in region_words) / 100.0,
                    4,
                ),
                token_count=len(region_words),
                rotation_degrees=rotation_degrees,
                quality_flags=quality_flags,
                evidence=evidence,
            )
        )
    return sorted(
        regions,
        key=lambda region: _region_sort_key(
            region,
            writing_mode=writing_mode,
            reading_order=reading_order,
        ),
    )


def _sort_region_words(
    words: list[PdfOcrWordEvidence],
    *,
    writing_mode: str,
) -> list[PdfOcrWordEvidence]:
    if writing_mode == "vertical":
        return sorted(words, key=lambda word: (word.top, -word.left))
    return sorted(words, key=lambda word: (word.top, word.left))


def _join_region_text(
    words: list[PdfOcrWordEvidence],
    *,
    writing_mode: str,
    raw: bool,
) -> str:
    values = [
        word.text if raw else word.normalized_text
        for word in words
        if (word.text if raw else word.normalized_text)
    ]
    if writing_mode == "vertical":
        return "".join(values)
    return " ".join(values)


def _region_sort_key(
    region: PdfOcrRegionEvidence,
    *,
    writing_mode: str,
    reading_order: str,
) -> tuple[int, int, int]:
    if region.region_type == "header":
        return (0, region.top, region.left)
    if region.region_type == "footer":
        return (2, region.top, region.left)
    if writing_mode == "vertical" and reading_order == "right_to_left":
        return (1, -region.left, region.top)
    return (1, region.top, region.left)


def _region_type(
    *,
    region_id: str,
    top: int,
    bottom: int,
    page_height: int | None,
) -> str:
    if region_id.endswith(":header"):
        return "header"
    if region_id.endswith(":footer"):
        return "footer"
    if not page_height:
        return "text"
    if bottom <= page_height * 0.08:
        return "header"
    if top >= page_height * 0.92:
        return "footer"
    return "text"


def _word_region_type(
    *,
    word: PdfOcrWordEvidence,
    page_height: int | None,
) -> str:
    if not page_height:
        return "text"
    bottom = word.top + word.height
    if bottom <= page_height * 0.08:
        return "header"
    if word.top >= page_height * 0.92:
        return "footer"
    return "text"


def _region_centers(values: list[int], region_count: int) -> list[float]:
    if region_count <= 1:
        return [float(values[0])] if values else [0.0]
    if len(values) <= region_count:
        return [float(value) for value in values]
    stride = max(1, len(values) // region_count)
    centers: list[float] = []
    for index in range(region_count):
        start = index * stride
        end = len(values) if index == region_count - 1 else min(len(values), start + stride)
        bucket = values[start:end] or [values[-1]]
        centers.append(statistics.mean(bucket))
    return centers


def _nearest_region_index(value: float, centers: list[float]) -> int:
    if not centers:
        return 0
    return min(range(len(centers)), key=lambda index: abs(value - centers[index]))


def _normalize_ocr_token(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ocr_word_quality_flags(*, text: str, confidence: float) -> list[str]:
    flags: list[str] = []
    normalized = _normalize_ocr_token(text)
    if confidence < 50:
        flags.append("low_confidence")
    if normalized and _symbol_ratio(normalized) >= 0.6:
        flags.append("symbol_heavy")
    if re.search(r"(.)\1{3,}", normalized):
        flags.append("repeated_garbage")
    return flags


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbol_count = sum(not character.isalnum() for character in text)
    return symbol_count / len(text)


def _cluster_count(values: list[int], gap: int) -> int:
    if not values:
        return 0
    clusters = 1
    previous = values[0]
    for value in values[1:]:
        if value - previous > gap:
            clusters += 1
        previous = value
    return clusters


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _geometry_left(item: dict[str, int] | PdfOcrWordEvidence) -> int:
    return item.left if isinstance(item, PdfOcrWordEvidence) else int(item["left"])


def _geometry_width(item: dict[str, int] | PdfOcrWordEvidence) -> int:
    return item.width if isinstance(item, PdfOcrWordEvidence) else int(item["width"])


def _geometry_height(item: dict[str, int] | PdfOcrWordEvidence) -> int:
    return item.height if isinstance(item, PdfOcrWordEvidence) else int(item["height"])
