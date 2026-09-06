from doc_loaders.pdf_page_evidence import analyze_ocr_tsv
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
    print("pdf page evidence checks passed")


if __name__ == "__main__":
    main()
