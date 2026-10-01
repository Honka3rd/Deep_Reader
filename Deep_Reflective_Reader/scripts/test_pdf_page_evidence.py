from doc_loaders.pdf_page_evidence import PdfPageLayoutEvidence, analyze_ocr_tsv
from document_structure.toc_detector import TableOfContentsDetector


def main() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t10\t20\t80\t90\t目錄\n"
        "5\t1\t1\t1\t2\t1\t200\t10\t20\t80\t90\t序\n"
        "5\t1\t1\t1\t3\t1\t300\t10\t20\t80\t90\t尾聲\n"
        "5\t1\t1\t1\t4\t1\t400\t10\t20\t80\t90\t讀後記\n"
        "5\t1\t1\t1\t5\t1\t500\t10\t20\t80\t90\t漂流船\n"
    )
    page = analyze_ocr_tsv(
        page_index=1,
        width=500,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )
    assert page.writing_mode == "vertical"
    assert page.reading_order == "right_to_left"
    assert page.orientation == "portrait"
    assert page.rotation_degrees == 0
    assert page.orientation_hypotheses[0]["rotation_degrees"] == 0
    assert page.writing_mode_confidence > 0.7
    assert page.reading_order_confidence > 0.7
    assert page.region_count >= 2
    candidate = TableOfContentsDetector().detect_page_candidates([page])[0]
    assert candidate.score >= 0.45
    assert "vertical_layout" in candidate.evidence
    detected = TableOfContentsDetector().detect_with_page_evidence(
        "正文没有可解析的目录文字",
        [page],
    )
    assert detected.detected
    assert detected.candidate_page_indices == [1]
    assert "page_layout_toc_candidates" in detected.reasons
    test_horizontal_toc_candidate_scores_without_marker()
    test_ocr_fragmented_title_candidate_scores_without_page_numbers()
    test_horizontal_geometry_reconstructs_toc_entries()
    test_vertical_rtl_geometry_reconstructs_toc_entries()
    test_horizontal_logical_reading_order_preserves_coordinates()
    test_vertical_logical_reading_order_preserves_coordinates()
    test_horizontal_orientation_and_reading_order_evidence()
    test_page_level_orientation_contract_serializes_region_count()
    test_coordinate_ocr_cost_policy_metadata()
    test_region_first_evidence_groups_vertical_columns()
    test_region_first_evidence_preserves_header_footer_regions()
    test_ocr_word_character_provenance_contract()
    test_low_quality_word_flags_are_preserved()
    test_empty_ocr_page_contract_keeps_zero_regions()
    test_ambiguous_orientation_retains_competing_hypotheses()
    test_toc_ocr_quality_gate_accepts_reliable_entries()
    test_toc_ocr_quality_gate_rejects_missing_page_numbers_without_inference()
    test_toc_ocr_quality_gate_rejects_non_monotonic_page_numbers()
    test_toc_ocr_quality_gate_rejects_ambiguous_orientation()
    test_artwork_only_page_is_not_toc_candidate()
    test_page_number_region_recognition_supports_multiple_systems()
    test_page_number_offset_validation_selects_unique_offset()
    test_page_number_offset_validation_rejects_ambiguous_offset()
    test_page_number_offset_validation_rejects_non_monotonic_numbers()
    print("pdf page evidence checks passed")


def test_horizontal_toc_candidate_scores_without_marker() -> None:
    page = PdfPageLayoutEvidence(
        page_index=0,
        width=1000,
        height=1400,
        native_text_chars=0,
        image_count=1,
        orientation="upright",
        writing_mode="horizontal",
        reading_order="left_to_right",
        ocr_confidence=82.0,
        ocr_text=(
            "Opening .... ①\n"
            "Storm .... ②\n"
            "Harbor .... ③\n"
            "Afterword .... ④\n"
            "1\n2\n3\n4"
        ),
        word_count=12,
        column_count=2,
        line_count=8,
        analysis_stage="coordinate_ocr",
    )

    candidate = TableOfContentsDetector().detect_page_candidates([page])[0]
    assert candidate.score >= 0.45
    assert "horizontal_layout" in candidate.evidence
    assert "dotted_leader_evidence" in candidate.evidence
    assert "page_number_evidence" in candidate.evidence
    assert "separated_page_number_column" in candidate.evidence
    assert "circled_or_boxed_page_number" in candidate.evidence


def test_ocr_fragmented_title_candidate_scores_without_page_numbers() -> None:
    page = PdfPageLayoutEvidence(
        page_index=0,
        width=900,
        height=1400,
        native_text_chars=0,
        image_count=1,
        orientation="upright",
        writing_mode="horizontal",
        reading_order="left_to_right",
        ocr_confidence=65.0,
        ocr_text="暗\n水\n幽\n靈\n序\n章\n終\n聲",
        word_count=8,
        column_count=1,
        line_count=8,
        analysis_stage="coordinate_ocr",
    )

    candidate = TableOfContentsDetector().detect_page_candidates([page])[0]
    assert candidate.score >= 0.45
    assert "short_title_density" in candidate.evidence
    assert "ocr_fragmented_title_density" in candidate.evidence


def test_horizontal_geometry_reconstructs_toc_entries() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t91\tOpening\n"
        "5\t1\t1\t1\t1\t2\t220\t100\t120\t20\t88\t....\n"
        "5\t1\t1\t1\t1\t3\t500\t100\t20\t20\t93\t1\n"
        "5\t1\t1\t1\t2\t1\t100\t140\t80\t20\t92\tStorm\n"
        "5\t1\t1\t1\t2\t2\t220\t140\t120\t20\t88\t....\n"
        "5\t1\t1\t1\t2\t3\t500\t140\t20\t20\t93\t2\n"
        "5\t1\t1\t1\t3\t1\t100\t180\t80\t20\t92\tHarbor\n"
        "5\t1\t1\t1\t3\t2\t220\t180\t120\t20\t88\t....\n"
        "5\t1\t1\t1\t3\t3\t500\t180\t20\t20\t93\t3\n"
    )
    page = analyze_ocr_tsv(
        page_index=0,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    entries = TableOfContentsDetector().reconstruct_page_entries(page)
    assert [entry.title for entry in entries] == ["Opening", "Storm", "Harbor"]
    assert [entry.page_number for entry in entries] == [1, 2, 3]
    assert all("geometry_horizontal_row" in entry.evidence for entry in entries)
    assert all("leader_line_endpoint" in entry.evidence for entry in entries)
    assert all(entry.title_box is not None for entry in entries)
    assert all(entry.page_number_box is not None for entry in entries)


def test_vertical_rtl_geometry_reconstructs_toc_entries() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t400\t100\t20\t50\t91\t暗\n"
        "5\t1\t1\t1\t2\t1\t400\t160\t20\t50\t91\t水\n"
        "5\t1\t1\t1\t3\t1\t400\t220\t20\t50\t91\t幽\n"
        "5\t1\t1\t1\t4\t1\t400\t280\t20\t50\t91\t靈\n"
        "5\t1\t1\t1\t5\t1\t400\t360\t20\t50\t91\t①\n"
        "5\t1\t1\t1\t6\t1\t300\t100\t20\t50\t91\t序\n"
        "5\t1\t1\t1\t7\t1\t300\t160\t20\t50\t91\t章\n"
        "5\t1\t1\t1\t8\t1\t300\t240\t20\t50\t91\t②\n"
    )
    page = analyze_ocr_tsv(
        page_index=0,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    entries = TableOfContentsDetector().reconstruct_page_entries(page)
    assert page.writing_mode == "vertical"
    assert page.reading_order == "right_to_left"
    assert [entry.title for entry in entries] == ["暗水幽靈", "序章"]
    assert [entry.page_number for entry in entries] == [1, 2]
    assert all("geometry_vertical_column" in entry.evidence for entry in entries)
    assert all("right_to_left_column_order" in entry.evidence for entry in entries)


def test_horizontal_orientation_and_reading_order_evidence() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])

    assert page.orientation == "portrait"
    assert page.rotation_degrees == 0
    assert {item["rotation_degrees"] for item in page.orientation_hypotheses} == {0, 90}
    assert page.writing_mode == "horizontal"
    assert page.reading_order == "left_to_right"
    assert page.writing_mode_confidence >= 0.6
    assert page.reading_order_confidence >= 0.7
    assert page.region_count >= 1
    assert "page_dimensions_orientation" in page.evidence


def test_page_level_orientation_contract_serializes_region_count() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])
    payload = page.to_dict()

    assert payload["width"] == 700
    assert payload["height"] == 1000
    assert payload["rotation_degrees"] == 0
    assert payload["orientation_hypotheses"]
    assert payload["writing_mode"] == "horizontal"
    assert payload["reading_order"] == "left_to_right"
    assert payload["region_count"] >= 1
    assert payload["ocr_confidence"] > 0
    assert payload["evidence_schema_version"] == 1


def test_coordinate_ocr_cost_policy_metadata() -> None:
    page = analyze_ocr_tsv(
        page_index=4,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=(
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t91\tOpening\n"
        ),
        source_sha256="abc123",
        ocr_language="eng+chi_sim",
        render_dpi=200,
        ocr_pass_count=1,
        cache_key="pdf-layout:v1:abc123:page:4:stage:coordinate_ocr:psm:11",
    )
    payload = page.to_dict()

    assert page.analysis_stage == "coordinate_ocr"
    assert page.analysis_cost_tier == "coordinate_ocr"
    assert page.ocr_engine == "tesseract"
    assert page.ocr_language == "eng+chi_sim"
    assert page.render_dpi == 200
    assert page.ocr_pass_count == 1
    assert page.cache_key.endswith(":stage:coordinate_ocr:psm:11")
    assert page.cache_hit is False
    assert page.failure_reason is None
    assert page.high_cost_analysis_permitted is False
    assert payload["analysis_cost_tier"] == "coordinate_ocr"
    assert payload["cache_hit"] is False


def test_region_first_evidence_groups_vertical_columns() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t400\t100\t20\t50\t91\t暗\n"
        "5\t1\t1\t1\t2\t1\t400\t160\t20\t50\t91\t水\n"
        "5\t1\t1\t1\t3\t1\t400\t220\t20\t50\t91\t幽\n"
        "5\t1\t1\t1\t4\t1\t400\t280\t20\t50\t91\t靈\n"
        "5\t1\t1\t1\t5\t1\t400\t360\t20\t50\t91\t①\n"
        "5\t1\t1\t1\t6\t1\t300\t100\t20\t50\t91\t序\n"
        "5\t1\t1\t1\t7\t1\t300\t160\t20\t50\t91\t章\n"
        "5\t1\t1\t1\t8\t1\t300\t240\t20\t50\t91\t②\n"
    )
    page = analyze_ocr_tsv(
        page_index=0,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    assert page.region_count == 2
    assert len(page.ocr_regions) == 2
    assert [region.left for region in page.ocr_regions] == [400, 300]
    assert page.ocr_regions[0].raw_ocr == "暗水幽靈①"
    assert page.ocr_regions[0].normalized_text == "暗水幽靈①"
    assert page.ocr_regions[0].writing_mode == "vertical"
    assert page.ocr_regions[0].reading_order == "right_to_left"
    assert page.ocr_regions[0].rotation_degrees == 0
    assert page.ocr_regions[0].confidence == 0.91
    assert page.ocr_regions[0].token_count == 5
    assert "vertical_region" in page.ocr_regions[0].evidence
    assert "right_to_left_region_order" in page.ocr_regions[0].evidence
    assert page.ocr_regions[0].right == 420
    assert page.ocr_regions[0].bottom == 410


def test_region_first_evidence_preserves_header_footer_regions() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t10\t80\t20\t91\tHeader\n"
        "5\t1\t1\t1\t2\t1\t100\t300\t80\t20\t92\tBody\n"
        "5\t1\t1\t1\t3\t1\t100\t940\t80\t20\t93\tFooter\n"
    )
    page = analyze_ocr_tsv(
        page_index=5,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )
    payload = page.to_dict()

    assert page.region_count == 3
    assert [region.region_type for region in page.ocr_regions] == [
        "header",
        "text",
        "footer",
    ]
    assert [region.raw_ocr for region in page.ocr_regions] == [
        "Header",
        "Body",
        "Footer",
    ]
    assert page.ocr_regions[0].region_id == "page:5:header"
    assert page.ocr_regions[-1].region_id == "page:5:footer"
    assert payload["ocr_regions"][0]["region_type"] == "header"
    assert payload["ocr_regions"][-1]["region_type"] == "footer"


def test_ocr_word_character_provenance_contract() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])
    payload = page.to_dict()
    first_word = page.ocr_words[0]
    first_payload = payload["ocr_words"][0]

    assert first_word.page_index == page.page_index
    assert first_word.region_id == f"page:{page.page_index}:region:0"
    assert first_word.normalized_text == first_word.text
    assert first_word.normalization_version == 1
    assert first_word.confidence > 0
    assert first_word.left >= 0
    assert first_word.top >= 0
    assert first_word.width > 0
    assert first_word.height > 0
    assert first_payload["page_index"] == page.page_index
    assert first_payload["region_id"] == first_word.region_id
    assert first_payload["normalized_text"] == first_word.normalized_text
    assert first_payload["normalization_version"] == 1
    assert first_payload["quality_flags"] == []


def test_low_quality_word_flags_are_preserved() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t49\t@@@@\n"
        "5\t1\t1\t1\t1\t2\t220\t100\t80\t20\t91\tTitle\n"
    )
    page = analyze_ocr_tsv(
        page_index=3,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    first_word = page.ocr_words[0]
    assert first_word.page_index == 3
    assert first_word.region_id == "page:3:region:0"
    assert first_word.normalized_text == "@@@@"
    assert first_word.normalization_version == 1
    assert "low_confidence" in first_word.quality_flags
    assert "symbol_heavy" in first_word.quality_flags
    assert "repeated_garbage" in first_word.quality_flags


def test_empty_ocr_page_contract_keeps_zero_regions() -> None:
    page = analyze_ocr_tsv(
        page_index=7,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=(
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        ),
    )

    assert page.region_count == 0
    assert page.word_count == 0
    assert page.orientation == "portrait"
    assert page.rotation_degrees == 0
    assert page.orientation_hypotheses
    assert page.evidence_schema_version == 1
    assert "no_ocr_words" in page.evidence


def test_ambiguous_orientation_retains_competing_hypotheses() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t91\tOpening\n"
        "5\t1\t1\t1\t1\t2\t220\t100\t120\t20\t88\t....\n"
        "5\t1\t1\t1\t1\t3\t500\t100\t20\t20\t93\t1\n"
    )
    page = analyze_ocr_tsv(
        page_index=0,
        width=1000,
        height=1020,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    assert page.orientation == "unknown"
    assert page.rotation_degrees is None
    assert "ambiguous_orientation" in page.evidence
    assert [item["orientation"] for item in page.orientation_hypotheses] == [
        "portrait",
        "landscape",
    ]
    assert {item["rotation_degrees"] for item in page.orientation_hypotheses} == {0, 90}
    assert all(item["confidence"] == 0.5 for item in page.orientation_hypotheses)


def test_horizontal_logical_reading_order_preserves_coordinates() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t91\tOpening\n"
        "5\t1\t1\t1\t1\t2\t220\t100\t120\t20\t88\t....\n"
        "5\t1\t1\t1\t1\t3\t500\t100\t20\t20\t93\t7\n"
    )
    page = analyze_ocr_tsv(
        page_index=4,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )
    detector = TableOfContentsDetector()

    tokens = detector.normalize_page_reading_order(page)
    assert [token.raw_text for token in tokens] == ["Opening", "....", "7"]
    assert [token.logical_index for token in tokens] == [0, 1, 2]
    assert all(token.page_index == 4 for token in tokens)
    assert tokens[0].box.left == 100
    assert tokens[0].box.right == 180
    assert tokens[0].writing_mode == "horizontal"
    assert tokens[0].order_hypothesis == "horizontal_ltr_rows"

    pair = detector.reconstruct_normalized_page_pairs(page)[0]
    assert pair.title == "Opening"
    assert pair.page_number == 7
    assert pair.raw_ocr == "Opening .... 7"
    assert pair.rotation == "portrait"
    assert pair.order_hypothesis == "horizontal_ltr_rows"
    assert pair.confidence_breakdown["leader_line_endpoint"] == 0.15
    assert pair.title_box is not None
    assert pair.page_number_box is not None


def test_vertical_logical_reading_order_preserves_coordinates() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t400\t100\t20\t50\t91\t暗\n"
        "5\t1\t1\t1\t2\t1\t400\t160\t20\t50\t91\t水\n"
        "5\t1\t1\t1\t3\t1\t400\t220\t20\t50\t91\t①\n"
        "5\t1\t1\t1\t4\t1\t300\t100\t20\t50\t91\t序\n"
        "5\t1\t1\t1\t5\t1\t300\t160\t20\t50\t91\t②\n"
    )
    page = analyze_ocr_tsv(
        page_index=2,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )
    detector = TableOfContentsDetector()

    tokens = detector.normalize_page_reading_order(page)
    assert [token.raw_text for token in tokens] == ["暗", "水", "①", "序", "②"]
    assert [token.group_index for token in tokens] == [0, 0, 0, 1, 1]
    assert tokens[0].writing_mode == "vertical"
    assert tokens[0].reading_order == "right_to_left"
    assert tokens[0].order_hypothesis == "vertical_rtl_columns_top_to_bottom"
    assert tokens[0].box.left == 400
    assert tokens[3].box.left == 300

    pairs = detector.reconstruct_normalized_page_pairs(page)
    assert [pair.title for pair in pairs] == ["暗水", "序"]
    assert [pair.page_number for pair in pairs] == [1, 2]
    assert pairs[0].raw_ocr == "暗水①"
    assert pairs[0].confidence_breakdown["right_to_left_column_order"] == 0.1
    assert pairs[0].order_hypothesis == "vertical_rtl_columns_top_to_bottom"


def test_toc_ocr_quality_gate_accepts_reliable_entries() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])

    quality = TableOfContentsDetector().evaluate_toc_ocr_quality(
        page,
        body_text="Opening\n正文\nStorm\n正文\nHarbor\n正文",
    )

    assert quality.detected
    assert quality.usable_for_splitting
    assert quality.entry_count == 3
    assert quality.title_completeness == 1.0
    assert quality.page_number_coverage == 1.0
    assert quality.geometric_pairing_coverage == 1.0
    assert quality.ordering_consistency == 1.0
    assert quality.body_title_recall == 1.0
    assert "toc_quality_usable_for_splitting" in quality.reasons


def test_toc_ocr_quality_gate_rejects_missing_page_numbers_without_inference() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t100\t80\t20\t91\tOpening\n"
        "5\t1\t1\t1\t2\t1\t100\t140\t80\t20\t92\tStorm\n"
        "5\t1\t1\t1\t3\t1\t100\t180\t80\t20\t92\tHarbor\n"
        "5\t1\t1\t1\t4\t1\t100\t220\t80\t20\t92\tAfterword\n"
        "5\t1\t1\t1\t5\t1\t100\t260\t80\t20\t92\tAppendix\n"
    )
    page = analyze_ocr_tsv(
        page_index=0,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text=tsv,
    )

    quality = TableOfContentsDetector().evaluate_toc_ocr_quality(
        page,
        body_text="Opening\nStorm\nHarbor\nAfterword\nAppendix",
    )

    assert quality.detected
    assert not quality.usable_for_splitting
    assert quality.entry_count == 0
    assert quality.page_number_coverage == 0.0
    assert "toc_quality_rejected_no_reconstructed_entries" in quality.reasons
    assert "toc_quality_rejected_missing_page_numbers" in quality.reasons
    assert "toc_quality_not_usable_for_splitting" in quality.reasons


def test_toc_ocr_quality_gate_rejects_non_monotonic_page_numbers() -> None:
    page = _horizontal_toc_page_with_numbers(["3", "2", "4"])

    quality = TableOfContentsDetector().evaluate_toc_ocr_quality(
        page,
        body_text="Opening\nStorm\nHarbor",
    )

    assert quality.detected
    assert not quality.usable_for_splitting
    assert quality.entry_count == 3
    assert quality.page_number_coverage == 1.0
    assert quality.ordering_consistency == 0.0
    assert "toc_quality_rejected_page_order" in quality.reasons
    assert "toc_quality_not_usable_for_splitting" in quality.reasons


def test_toc_ocr_quality_gate_rejects_ambiguous_orientation() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])
    page = PdfPageLayoutEvidence(
        page_index=page.page_index,
        width=page.width,
        height=page.height,
        native_text_chars=page.native_text_chars,
        image_count=page.image_count,
        orientation="unknown",
        writing_mode=page.writing_mode,
        reading_order=page.reading_order,
        ocr_confidence=page.ocr_confidence,
        ocr_text=page.ocr_text,
        word_count=page.word_count,
        column_count=page.column_count,
        line_count=page.line_count,
        analysis_stage=page.analysis_stage,
        evidence=["ambiguous_orientation"],
        ocr_words=page.ocr_words,
    )

    quality = TableOfContentsDetector().evaluate_toc_ocr_quality(
        page,
        body_text="Opening\n正文\nStorm\n正文\nHarbor\n正文",
    )

    assert quality.detected
    assert not quality.usable_for_splitting
    assert quality.entry_count == 3
    assert quality.page_number_coverage == 1.0
    assert quality.ordering_consistency == 1.0
    assert "toc_quality_rejected_ambiguous_orientation" in quality.reasons
    assert "toc_quality_not_usable_for_splitting" in quality.reasons


def test_artwork_only_page_is_not_toc_candidate() -> None:
    page = PdfPageLayoutEvidence(
        page_index=0,
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
    )
    detector = TableOfContentsDetector()

    assert detector.detect_page_candidates([page]) == []
    quality = detector.evaluate_toc_ocr_quality(page, body_text="Body text")

    assert not quality.detected
    assert not quality.usable_for_splitting
    assert quality.entry_count == 0
    assert "toc_quality_rejected_not_detected" in quality.reasons
    assert "toc_quality_rejected_no_reconstructed_entries" in quality.reasons
    assert "toc_quality_not_usable_for_splitting" in quality.reasons


def test_page_number_region_recognition_supports_multiple_systems() -> None:
    page = _horizontal_toc_page_with_titles_and_numbers(
        [
            ("Arabic", "12"),
            ("Chinese", "二十三"),
            ("Roman", "IX"),
            ("Circled", "④"),
            ("Boxed", "【15】"),
            ("Fragmented", "1 6"),
        ]
    )

    regions = TableOfContentsDetector().recognize_page_number_regions(page)

    assert [region.value for region in regions] == [12, 23, 9, 4, 15, 16]
    assert [region.numeral_system for region in regions] == [
        "arabic",
        "chinese",
        "roman",
        "circled",
        "boxed_arabic",
        "arabic",
    ]
    assert all(region.box is not None for region in regions)
    assert any("fragmented_page_number" in region.evidence for region in regions)
    assert all("region_scoped_page_number" in region.evidence for region in regions)


def test_page_number_offset_validation_selects_unique_offset() -> None:
    page = _horizontal_toc_page_with_numbers(["1", "2", "3"])
    pairs = TableOfContentsDetector().reconstruct_normalized_page_pairs(page)

    result = TableOfContentsDetector().validate_page_number_anchor_offsets(
        pairs,
        {
            0: "Opening body",
            1: "Storm body",
            2: "Harbor body",
            3: "Appendix",
        },
    )

    assert result.valid
    assert result.selected_offset == -1
    assert result.matched_entry_count == 3
    assert "page_number_offset_validated" in result.reasons


def test_page_number_offset_validation_rejects_ambiguous_offset() -> None:
    page = _horizontal_toc_page_with_titles_and_numbers(
        [("Opening", "1"), ("Storm", "2")]
    )
    pairs = TableOfContentsDetector().reconstruct_normalized_page_pairs(page)

    result = TableOfContentsDetector().validate_page_number_anchor_offsets(
        pairs,
        {
            0: "Opening",
            1: "Storm",
            2: "Opening",
            3: "Storm",
        },
    )

    assert not result.valid
    assert result.selected_offset is None
    assert "ambiguous_page_number_offset" in result.reasons
    assert "page_number_offset_rejected" in result.reasons


def test_page_number_offset_validation_rejects_non_monotonic_numbers() -> None:
    page = _horizontal_toc_page_with_numbers(["3", "2", "4"])
    pairs = TableOfContentsDetector().reconstruct_normalized_page_pairs(page)

    result = TableOfContentsDetector().validate_page_number_anchor_offsets(
        pairs,
        {
            1: "Storm",
            2: "Opening",
            3: "Harbor",
        },
    )

    assert not result.valid
    assert result.selected_offset is None
    assert "non_monotonic_page_numbers" in result.reasons
    assert "page_number_offset_rejected" in result.reasons


def _horizontal_toc_page_with_numbers(numbers: list[str]) -> PdfPageLayoutEvidence:
    return _horizontal_toc_page_with_titles_and_numbers(
        list(zip(["Opening", "Storm", "Harbor"], numbers))
    )


def _horizontal_toc_page_with_titles_and_numbers(
    entries: list[tuple[str, str]],
) -> PdfPageLayoutEvidence:
    rows = [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    ]
    for index, (title, number) in enumerate(entries, start=1):
        top = 80 + index * 40
        rows.extend(
            [
                f"5\t1\t1\t1\t{index}\t1\t100\t{top}\t80\t20\t91\t{title}",
                f"5\t1\t1\t1\t{index}\t2\t220\t{top}\t120\t20\t88\t....",
                f"5\t1\t1\t1\t{index}\t3\t500\t{top}\t20\t20\t93\t{number}",
            ]
        )
    return analyze_ocr_tsv(
        page_index=0,
        width=700,
        height=1000,
        native_text_chars=0,
        image_count=1,
        tsv_text="\n".join(rows) + "\n",
    )


if __name__ == "__main__":
    main()
