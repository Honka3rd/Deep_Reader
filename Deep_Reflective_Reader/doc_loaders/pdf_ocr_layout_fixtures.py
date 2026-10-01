"""Synthetic OCR layout fixtures for PDF layout regression tests.

These fixtures describe expected loader-level evidence only. They do not
authorize hierarchy splitting and they do not replace real PDF smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pdf_page_evidence import PdfPageLayoutEvidence, analyze_ocr_tsv


TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext"
)


@dataclass(frozen=True)
class PdfOcrLayoutFixture:
    fixture_id: str
    source_doc_name: str
    layout_kind: str
    width: int
    height: int
    tsv_rows: tuple[str, ...] = ()
    image_count: int = 1
    native_text_chars: int = 0
    expected_orientation: str = "portrait"
    expected_writing_mode: str = "horizontal"
    expected_reading_order: str = "left_to_right"
    expected_region_types: tuple[str, ...] = ("text",)
    expected_min_region_count: int = 1
    expected_toc_detected: bool = False
    expected_usable_for_splitting: bool = False
    expected_entry_titles: tuple[str, ...] = ()
    expected_page_numbers: tuple[int, ...] = ()
    expected_evidence: tuple[str, ...] = ()
    expected_limitations: tuple[str, ...] = ()
    expected_fallback_reasons: tuple[str, ...] = ()
    renderer_failure_detail: str | None = None

    def tsv_text(self) -> str:
        if not self.tsv_rows:
            return TSV_HEADER + "\n"
        return "\n".join((TSV_HEADER, *self.tsv_rows)) + "\n"

    def build_page_evidence(self, *, page_index: int = 0) -> PdfPageLayoutEvidence:
        return analyze_ocr_tsv(
            page_index=page_index,
            width=self.width,
            height=self.height,
            native_text_chars=self.native_text_chars,
            image_count=self.image_count,
            tsv_text=self.tsv_text(),
            source_file_name=f"{self.source_doc_name}.pdf",
            source_sha256=f"synthetic-fixture:{self.fixture_id}",
        )


def _row(
    *,
    line: int,
    word: int,
    left: int,
    top: int,
    width: int,
    height: int,
    conf: int,
    text: str,
) -> str:
    return f"5\t1\t1\t1\t{line}\t{word}\t{left}\t{top}\t{width}\t{height}\t{conf}\t{text}"


OCR_LAYOUT_FIXTURES: dict[str, PdfOcrLayoutFixture] = {
    "dark_water_artistic_vertical_toc": PdfOcrLayoutFixture(
        fixture_id="dark_water_artistic_vertical_toc",
        source_doc_name="暗水幽灵",
        layout_kind="artistic_vertical_toc",
        width=700,
        height=1000,
        tsv_rows=(
            _row(line=1, word=1, left=440, top=110, width=20, height=50, conf=91, text="暗"),
            _row(line=2, word=1, left=440, top=170, width=20, height=50, conf=91, text="水"),
            _row(line=3, word=1, left=440, top=230, width=20, height=50, conf=91, text="幽"),
            _row(line=4, word=1, left=440, top=290, width=20, height=50, conf=91, text="靈"),
            _row(line=5, word=1, left=440, top=370, width=20, height=50, conf=91, text="①"),
            _row(line=6, word=1, left=320, top=110, width=20, height=50, conf=91, text="序"),
            _row(line=7, word=1, left=320, top=170, width=20, height=50, conf=91, text="章"),
            _row(line=8, word=1, left=320, top=250, width=20, height=50, conf=91, text="②"),
        ),
        expected_writing_mode="vertical",
        expected_reading_order="right_to_left",
        expected_min_region_count=2,
        expected_toc_detected=True,
        expected_usable_for_splitting=False,
        expected_entry_titles=("暗水幽靈", "序章"),
        expected_page_numbers=(1, 2),
        expected_evidence=("vertical_layout", "right_to_left_column_order"),
        expected_limitations=("artistic_scan_not_full_text_equal",),
    ),
    "guo_fu_lun_vertical_body_text": PdfOcrLayoutFixture(
        fixture_id="guo_fu_lun_vertical_body_text",
        source_doc_name="國富論",
        layout_kind="vertical_body_text",
        width=760,
        height=1080,
        tsv_rows=(
            _row(line=1, word=1, left=520, top=120, width=20, height=52, conf=89, text="分"),
            _row(line=2, word=1, left=520, top=180, width=20, height=52, conf=89, text="工"),
            _row(line=3, word=1, left=520, top=240, width=20, height=52, conf=89, text="市"),
            _row(line=4, word=1, left=520, top=300, width=20, height=52, conf=89, text="場"),
            _row(line=5, word=1, left=400, top=120, width=20, height=52, conf=90, text="價"),
            _row(line=6, word=1, left=400, top=180, width=20, height=52, conf=90, text="格"),
            _row(line=7, word=1, left=400, top=240, width=20, height=52, conf=90, text="勞"),
            _row(line=8, word=1, left=400, top=300, width=20, height=52, conf=90, text="動"),
        ),
        expected_writing_mode="vertical",
        expected_reading_order="right_to_left",
        expected_min_region_count=2,
        expected_toc_detected=False,
        expected_fallback_reasons=("vertical_body_text_without_page_numbers",),
    ),
    "horizontal_scan_leader_lines": PdfOcrLayoutFixture(
        fixture_id="horizontal_scan_leader_lines",
        source_doc_name="horizontal_scan",
        layout_kind="horizontal_toc_leaders",
        width=900,
        height=1200,
        tsv_rows=(
            _row(line=1, word=1, left=120, top=150, width=100, height=24, conf=92, text="Opening"),
            _row(line=1, word=2, left=320, top=150, width=140, height=18, conf=88, text="...."),
            _row(line=1, word=3, left=650, top=150, width=24, height=24, conf=93, text="1"),
            _row(line=2, word=1, left=120, top=200, width=80, height=24, conf=92, text="Storm"),
            _row(line=2, word=2, left=320, top=200, width=140, height=18, conf=88, text="...."),
            _row(line=2, word=3, left=650, top=200, width=24, height=24, conf=93, text="2"),
            _row(line=3, word=1, left=120, top=250, width=90, height=24, conf=92, text="Harbor"),
            _row(line=3, word=2, left=320, top=250, width=140, height=18, conf=88, text="...."),
            _row(line=3, word=3, left=650, top=250, width=24, height=24, conf=93, text="3"),
        ),
        expected_toc_detected=True,
        expected_usable_for_splitting=True,
        expected_entry_titles=("Opening", "Storm", "Harbor"),
        expected_page_numbers=(1, 2, 3),
        expected_evidence=("horizontal_layout", "leader_line_endpoint"),
    ),
    "mixed_orientation_regions": PdfOcrLayoutFixture(
        fixture_id="mixed_orientation_regions",
        source_doc_name="mixed_orientation",
        layout_kind="mixed_header_vertical_body_footer",
        width=720,
        height=1000,
        tsv_rows=(
            _row(line=1, word=1, left=100, top=20, width=110, height=24, conf=90, text="Header"),
            _row(line=2, word=1, left=430, top=130, width=20, height=54, conf=91, text="第"),
            _row(line=3, word=1, left=430, top=190, width=20, height=54, conf=91, text="一"),
            _row(line=4, word=1, left=430, top=250, width=20, height=54, conf=91, text="章"),
            _row(line=5, word=1, left=300, top=130, width=20, height=54, conf=91, text="正"),
            _row(line=6, word=1, left=300, top=190, width=20, height=54, conf=91, text="文"),
            _row(line=7, word=1, left=100, top=940, width=100, height=24, conf=90, text="Footer"),
        ),
        expected_writing_mode="vertical",
        expected_reading_order="right_to_left",
        expected_region_types=("header", "text", "footer"),
        expected_min_region_count=3,
        expected_toc_detected=False,
        expected_fallback_reasons=("mixed_layout_requires_region_evidence",),
    ),
    "circular_page_numbers": PdfOcrLayoutFixture(
        fixture_id="circular_page_numbers",
        source_doc_name="circular_page_numbers",
        layout_kind="horizontal_toc_circled_numbers",
        width=900,
        height=1200,
        tsv_rows=(
            _row(line=1, word=1, left=120, top=150, width=100, height=24, conf=92, text="First"),
            _row(line=1, word=2, left=650, top=150, width=24, height=24, conf=93, text="①"),
            _row(line=2, word=1, left=120, top=200, width=100, height=24, conf=92, text="Second"),
            _row(line=2, word=2, left=650, top=200, width=24, height=24, conf=93, text="②"),
            _row(line=3, word=1, left=120, top=250, width=100, height=24, conf=92, text="Third"),
            _row(line=3, word=2, left=650, top=250, width=24, height=24, conf=93, text="③"),
        ),
        expected_toc_detected=True,
        expected_usable_for_splitting=True,
        expected_entry_titles=("First", "Second", "Third"),
        expected_page_numbers=(1, 2, 3),
        expected_evidence=("circled_or_boxed_page_number",),
    ),
    "renderer_decode_failure": PdfOcrLayoutFixture(
        fixture_id="renderer_decode_failure",
        source_doc_name="renderer_decode_failure",
        layout_kind="renderer_failure",
        width=700,
        height=1000,
        tsv_rows=(),
        expected_region_types=(),
        expected_min_region_count=0,
        expected_toc_detected=False,
        expected_fallback_reasons=("pdf_render_failed", "layout_evidence_empty"),
        renderer_failure_detail="pdf_render_failed:JBIG2Decode",
    ),
}


def iter_ocr_layout_fixtures() -> tuple[PdfOcrLayoutFixture, ...]:
    return tuple(OCR_LAYOUT_FIXTURES.values())
