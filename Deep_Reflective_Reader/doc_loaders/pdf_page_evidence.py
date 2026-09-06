"""Page-level PDF layout evidence used by deterministic structure analysis."""

from __future__ import annotations

import csv
import io
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PdfPageTextBoundary:
    """Exact raw-text span belonging to one PDF page in canonical page order."""

    page_index: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class PdfPageLayoutEvidence:
    page_index: int
    width: int | None
    height: int | None
    native_text_chars: int
    image_count: int
    orientation: str = "unknown"
    writing_mode: str = "unknown"
    reading_order: str = "unknown"
    ocr_confidence: float = 0.0
    ocr_text: str = ""
    word_count: int = 0
    column_count: int = 0
    line_count: int = 0
    analysis_stage: str = "inventory"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "native_text_chars": self.native_text_chars,
            "image_count": self.image_count,
            "orientation": self.orientation,
            "writing_mode": self.writing_mode,
            "reading_order": self.reading_order,
            "ocr_confidence": self.ocr_confidence,
            "ocr_text": self.ocr_text,
            "word_count": self.word_count,
            "column_count": self.column_count,
            "line_count": self.line_count,
            "analysis_stage": self.analysis_stage,
            "evidence": list(self.evidence),
        }


def analyze_ocr_tsv(
    *,
    page_index: int,
    width: int | None,
    height: int | None,
    native_text_chars: int,
    image_count: int,
    tsv_text: str,
) -> PdfPageLayoutEvidence:
    """Convert Tesseract TSV into compact layout hypotheses."""
    rows = list(csv.DictReader(io.StringIO(tsv_text), delimiter="\t"))
    words: list[dict[str, object]] = []
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
            {
                "text": text,
                "confidence": confidence,
                "left": left,
                "top": top,
                "width": word_width,
                "height": word_height,
                "line": row.get("line_num", ""),
            }
        )

    if not words:
        return PdfPageLayoutEvidence(
            page_index=page_index,
            width=width,
            height=height,
            native_text_chars=native_text_chars,
            image_count=image_count,
            analysis_stage="coordinate_ocr",
            evidence=["no_ocr_words"],
        )

    geometry = line_boxes or words
    median_width = statistics.median(int(item["width"]) for item in geometry)
    median_height = statistics.median(int(item["height"]) for item in geometry)
    x_centers = sorted(
        {round(int(item["left"]) + int(item["width"]) / 2) for item in geometry}
    )
    line_ids = (
        {str(word["line"]) for word in words if str(word["line"])}
        if not line_boxes
        else {str(index) for index in range(len(line_boxes))}
    )
    vertical_shape = median_height > median_width * 1.35
    column_count = _cluster_count(x_centers, max(24, int(median_width * 1.5)))
    writing_mode = "vertical" if vertical_shape and column_count >= 2 else "horizontal"
    reading_order = "right_to_left" if writing_mode == "vertical" else "left_to_right"
    text = "\n".join(str(word["text"]) for word in words)
    confidence = statistics.mean(float(word["confidence"]) for word in words) / 100.0
    evidence = [
        "coordinate_ocr",
        "vertical_glyph_geometry" if vertical_shape else "horizontal_glyph_geometry",
    ]
    if column_count >= 2:
        evidence.append("multiple_text_columns")
    return PdfPageLayoutEvidence(
        page_index=page_index,
        width=width,
        height=height,
        native_text_chars=native_text_chars,
        image_count=image_count,
        orientation="portrait" if (height or 0) >= (width or 0) else "landscape",
        writing_mode=writing_mode,
        reading_order=reading_order,
        ocr_confidence=round(confidence, 4),
        ocr_text=text,
        word_count=len(words),
        column_count=column_count,
        line_count=len(line_ids),
        analysis_stage="coordinate_ocr",
        evidence=evidence,
    )


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
