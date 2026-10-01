#!/usr/bin/env python3
"""Regression checks for synthetic PDF OCR layout fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.pdf_ocr_layout_fixtures import (  # noqa: E402
    OCR_LAYOUT_FIXTURES,
    PdfOcrLayoutFixture,
    iter_ocr_layout_fixtures,
)
from document_structure.toc_detector import TableOfContentsDetector  # noqa: E402


REQUIRED_FIXTURE_IDS = {
    "dark_water_artistic_vertical_toc",
    "guo_fu_lun_vertical_body_text",
    "horizontal_scan_leader_lines",
    "mixed_orientation_regions",
    "circular_page_numbers",
    "renderer_decode_failure",
}


def main() -> None:
    test_fixture_catalog_covers_required_pdf_ocr_cases()
    test_layout_fixtures_match_expected_page_evidence()
    test_renderer_failure_fixture_requires_conservative_fallback()
    print("pdf OCR layout fixture checks passed")


def test_fixture_catalog_covers_required_pdf_ocr_cases() -> None:
    fixture_ids = set(OCR_LAYOUT_FIXTURES)

    assert REQUIRED_FIXTURE_IDS <= fixture_ids
    assert OCR_LAYOUT_FIXTURES["dark_water_artistic_vertical_toc"].source_doc_name == "暗水幽灵"
    assert OCR_LAYOUT_FIXTURES["guo_fu_lun_vertical_body_text"].source_doc_name == "國富論"
    assert (
        OCR_LAYOUT_FIXTURES["renderer_decode_failure"].renderer_failure_detail
        == "pdf_render_failed:JBIG2Decode"
    )


def test_layout_fixtures_match_expected_page_evidence() -> None:
    detector = TableOfContentsDetector()
    for fixture in iter_ocr_layout_fixtures():
        if fixture.renderer_failure_detail:
            continue
        page = fixture.build_page_evidence()
        candidates = detector.detect_page_candidates([page])
        candidate = candidates[0] if candidates else None
        entries = detector.reconstruct_page_entries(page)
        quality = detector.evaluate_toc_ocr_quality(
            page,
            body_text="Opening Storm First Second Third 分工市場價格勞動 正文",
        )
        actual_region_types = {region.region_type for region in page.ocr_regions}
        expected_region_types = set(fixture.expected_region_types)
        actual_evidence = set(page.evidence)
        if candidate is not None:
            actual_evidence.update(candidate.evidence)
        for entry in entries:
            actual_evidence.update(entry.evidence)
        actual_evidence.update(quality.reasons)

        assert page.orientation == fixture.expected_orientation, fixture.fixture_id
        assert page.writing_mode == fixture.expected_writing_mode, fixture.fixture_id
        assert page.reading_order == fixture.expected_reading_order, fixture.fixture_id
        assert page.region_count >= fixture.expected_min_region_count, fixture.fixture_id
        assert expected_region_types <= actual_region_types, fixture.fixture_id

        if fixture.expected_toc_detected:
            assert candidate is not None, fixture.fixture_id
            assert (
                quality.usable_for_splitting
                is fixture.expected_usable_for_splitting
            ), fixture.fixture_id
            assert tuple(entry.title for entry in entries) == fixture.expected_entry_titles
            assert tuple(entry.page_number for entry in entries) == fixture.expected_page_numbers
        else:
            assert not quality.usable_for_splitting, fixture.fixture_id

        for expected in fixture.expected_evidence:
            assert expected in actual_evidence, f"{fixture.fixture_id}: missing {expected}"


def test_renderer_failure_fixture_requires_conservative_fallback() -> None:
    fixture = OCR_LAYOUT_FIXTURES["renderer_decode_failure"]
    page = fixture.build_page_evidence()

    assert isinstance(fixture, PdfOcrLayoutFixture)
    assert fixture.renderer_failure_detail == "pdf_render_failed:JBIG2Decode"
    assert fixture.expected_fallback_reasons == (
        "pdf_render_failed",
        "layout_evidence_empty",
    )
    assert page.word_count == 0
    assert page.region_count == 0
    assert "no_ocr_words" in page.evidence


if __name__ == "__main__":
    main()
