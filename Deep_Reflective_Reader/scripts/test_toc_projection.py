from doc_loaders.pdf_page_evidence import PdfPageLayoutEvidence, PdfPageTextBoundary
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode


def _page(index: int, text: str) -> PdfPageLayoutEvidence:
    return PdfPageLayoutEvidence(
        page_index=index,
        width=1000,
        height=1600,
        native_text_chars=0,
        image_count=1,
        writing_mode="vertical",
        reading_order="right_to_left",
        ocr_text=text,
        word_count=20,
        column_count=4,
        line_count=8,
        analysis_stage="coordinate_ocr",
    )


def main() -> None:
    raw_text = (
        "目录\n1. Intro  1\n2. Body  1\n3. End  1\n\n"
        "1. Intro\nIntro text.\n2. Body\nBody text.\n3. End\nEnd text."
    )
    pages = [_page(0, "目录 Intro Body End"), _page(1, "目录 Intro Body End")]
    boundaries = [
        PdfPageTextBoundary(page_index=0, char_start=0, char_end=len(raw_text), text=raw_text),
    ]
    document = StructuredDocumentBuilder().build(
        document_id="demo",
        title="demo",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )
    assert document.parse_provenance["toc_detection"]["usable"]
    assert document.parse_provenance["effective_parser"] == "validated_toc_projection"
    assert len(document.chapters) == 3
    assert all(chapter.sections for chapter in document.chapters)

    untrusted = StructuredDocumentBuilder().build(
        document_id="fallback",
        title="fallback",
        raw_text="目录\nIntro\nBody\nEnd\n正文没有对应标题",
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )
    assert untrusted.parse_provenance["toc_detection"]["usable"] is False
    assert "effective_parser" not in untrusted.parse_provenance
    test_deep_toc_levels_collapse_to_two_layer_hierarchy()
    test_inconsistent_toc_page_offset_falls_back_atomically()
    test_layout_toc_without_page_anchors_falls_back_with_provenance()
    test_multi_page_toc_grouping_records_page_span()
    test_multi_page_toc_grouping_terminates_on_artwork_page()
    print("toc projection checks passed")


def test_deep_toc_levels_collapse_to_two_layer_hierarchy() -> None:
    raw_text = (
        "目录\n"
        "1. Intro  1\n"
        "1.1 Scope  1\n"
        "1.1.1 Details  1\n"
        "1.2 Other  1\n"
        "2. End  1\n\n"
        "1. Intro\n"
        "Intro text.\n"
        "1.1 Scope\n"
        "Scope text.\n"
        "1.1.1 Details\n"
        "Details text.\n"
        "1.2 Other\n"
        "Other text.\n"
        "2. End\n"
        "End text."
    )
    pages = [_page(0, "目录 Intro Scope Details Other End"), _page(1, "目录 Intro Scope Details Other End")]
    boundaries = [
        PdfPageTextBoundary(page_index=0, char_start=0, char_end=len(raw_text), text=raw_text),
    ]

    document = StructuredDocumentBuilder().build(
        document_id="deep-toc",
        title="deep-toc",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )

    assert document.parse_provenance["toc_detection"]["shape"] == "deep_hierarchy"
    assert document.parse_provenance["effective_parser"] == "validated_toc_projection"
    assert "toc_projection" in document.parse_provenance
    projection_entries = document.parse_provenance["toc_projection"]["entries"]
    assert any(
        entry["title"] == "1.1.1 Details"
        and entry["included_as_section"] is False
        and entry["merge_reason"] == "deeper_than_two_collapsed_into_previous_section"
        for entry in projection_entries
    )

    assert [chapter.title for chapter in document.chapters] == ["1. Intro", "2. End"]
    intro_sections = document.chapters[0].sections
    assert [section.title for section in intro_sections] == [
        "1. Intro",
        "1.1 Scope",
        "1.2 Other",
    ]
    assert all(section.level <= 2 for chapter in document.chapters for section in chapter.sections)
    assert "1.1.1 Details" in intro_sections[1].content
    assert "Details text." in intro_sections[1].content
    assert "1.1.1 Details" not in [section.title for section in intro_sections]
    payload = document.to_dict()
    assert "sections" not in payload
    assert "structure_nodes" not in payload


def test_inconsistent_toc_page_offset_falls_back_atomically() -> None:
    page_1 = (
        "目录\n"
        "1. Intro  1\n"
        "2. Body  2\n"
        "3. End  3\n\n"
        "1. Intro\n"
        "Intro text.\n"
        "2. Body\n"
        "Body text.\n"
        "3. End\n"
        "End text."
    )
    page_2 = "Appendix marker only."
    raw_text = "\n\n".join([page_1, page_2])
    page_2_start = len(page_1) + 2
    pages = [_page(0, "目录 Intro Body End"), _page(1, "目录 Intro Body End")]
    boundaries = [
        PdfPageTextBoundary(
            page_index=0,
            char_start=0,
            char_end=len(page_1),
            text=page_1,
        ),
        PdfPageTextBoundary(
            page_index=1,
            char_start=page_2_start,
            char_end=len(raw_text),
            text=page_2,
        ),
    ]

    document = StructuredDocumentBuilder().build(
        document_id="bad-offset-toc",
        title="bad-offset-toc",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )

    assert document.parse_provenance["toc_detection"]["usable"] is True
    assert "effective_parser" not in document.parse_provenance
    assert "toc_projection" not in document.parse_provenance
    assert len(document.chapters) != 3
    payload = document.to_dict()
    assert "sections" not in payload
    assert "structure_nodes" not in payload


def test_layout_toc_without_page_anchors_falls_back_with_provenance() -> None:
    raw_text = (
        "Body introduction without a table of contents marker.\n"
        "Opening body.\n"
        "Storm body.\n"
        "Harbor body."
    )
    pages = [
        PdfPageLayoutEvidence(
            page_index=0,
            width=1000,
            height=1400,
            native_text_chars=0,
            image_count=1,
            orientation="portrait",
            writing_mode="horizontal",
            reading_order="left_to_right",
            ocr_confidence=80.0,
            ocr_text="Opening\nStorm\nHarbor\nAfterword\nAppendix",
            word_count=5,
            column_count=1,
            line_count=5,
            analysis_stage="coordinate_ocr",
        ),
        PdfPageLayoutEvidence(
            page_index=1,
            width=1000,
            height=1400,
            native_text_chars=0,
            image_count=1,
            orientation="portrait",
            writing_mode="horizontal",
            reading_order="left_to_right",
            ocr_confidence=80.0,
            ocr_text="Opening body Storm body Harbor body",
            word_count=6,
            column_count=1,
            line_count=3,
            analysis_stage="coordinate_ocr",
        ),
    ]
    boundaries = [
        PdfPageTextBoundary(
            page_index=0,
            char_start=0,
            char_end=len(raw_text),
            text=raw_text,
        )
    ]

    document = StructuredDocumentBuilder().build(
        document_id="layout-toc-without-page-anchors",
        title="layout-toc-without-page-anchors",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )

    toc_detection = document.parse_provenance["toc_detection"]
    assert toc_detection["detected"] is True
    assert toc_detection["usable"] is False
    assert "page_layout_toc_candidates" in toc_detection["reasons"]
    assert "toc_projection_rejected_missing_page_numbers" in toc_detection["reasons"]
    assert "toc_projection_rejected_global_validation" in toc_detection["reasons"]
    assert "effective_parser" not in document.parse_provenance
    assert "toc_projection" not in document.parse_provenance
    payload = document.to_dict()
    assert "sections" not in payload
    assert "structure_nodes" not in payload


def test_multi_page_toc_grouping_records_page_span() -> None:
    page_1 = "目录\n1. Intro  1\n2. Body  1\n3. End  1\n\n"
    page_2 = "4. Appendix  1\n\n"
    body = (
        "1. Intro\nIntro text.\n"
        "2. Body\nBody text.\n"
        "3. End\nEnd text.\n"
        "4. Appendix\nAppendix text."
    )
    raw_text = page_1 + page_2 + body
    pages = [_page(0, "目录 Intro Body End"), _page(1, "Appendix")]
    boundaries = [
        PdfPageTextBoundary(page_index=0, char_start=0, char_end=len(raw_text), text=raw_text),
    ]

    document = StructuredDocumentBuilder().build(
        document_id="multi-page-toc-group",
        title="multi-page-toc-group",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )

    toc_detection = document.parse_provenance["toc_detection"]
    assert toc_detection["usable"] is True
    assert "toc_projection_page_group:0-1" in toc_detection["reasons"]
    assert document.parse_provenance["effective_parser"] == "validated_toc_projection"
    assert "toc_projection" in document.parse_provenance


def test_multi_page_toc_grouping_terminates_on_artwork_page() -> None:
    raw_text = (
        "目录\n"
        "1. Intro  1\n"
        "2. Body  1\n"
        "3. End  1\n\n"
        "1. Intro\nIntro text.\n"
        "2. Body\nBody text.\n"
        "3. End\nEnd text."
    )
    pages = [
        _page(0, "目录 Intro Body"),
        PdfPageLayoutEvidence(
            page_index=1,
            width=1000,
            height=1400,
            native_text_chars=0,
            image_count=1,
            orientation="portrait",
            writing_mode="unknown",
            reading_order="unknown",
            ocr_confidence=0.0,
            ocr_text="",
            word_count=0,
            column_count=0,
            line_count=0,
            analysis_stage="inventory",
            evidence=["artwork_only_page"],
        ),
        _page(2, "End Appendix"),
    ]
    boundaries = [
        PdfPageTextBoundary(page_index=0, char_start=0, char_end=len(raw_text), text=raw_text),
    ]

    document = StructuredDocumentBuilder().build(
        document_id="toc-group-artwork-break",
        title="toc-group-artwork-break",
        raw_text=raw_text,
        language=LanguageCode.EN,
        page_evidence=pages,
        page_boundaries=boundaries,
    )

    toc_detection = document.parse_provenance["toc_detection"]
    assert toc_detection["usable"] is False
    assert "toc_projection_rejected_missing_page_group" in toc_detection["reasons"]
    assert "effective_parser" not in document.parse_provenance
    assert "toc_projection" not in document.parse_provenance


if __name__ == "__main__":
    main()
