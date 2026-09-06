from document_structure.text_normalization import normalize_ocr_text, normalize_ocr_whitespace
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.toc_detector import TableOfContentsDetector, TocShape


def main() -> None:
    assert normalize_ocr_whitespace("海 底 沉 睡 的 森 林") == "海底沉睡的森林"
    assert normalize_ocr_whitespace("“ 那 当 然 。”") == "“那当然。”"
    assert normalize_ocr_whitespace("Dark Water") == "Dark Water"
    assert normalize_ocr_whitespace("第 1 章: Dark Water") == "第1章: Dark Water"
    assert normalize_ocr_text("ＡＢＣ　海 底") == "ABC海底"
    assert LLMSectionSplitter._normalize_line("海 底 沉 睡 的 森 林") == "海底沉睡的森林"
    toc = TableOfContentsDetector().detect(
        "Table of Contents\n"
        "Chapter 1 ........................ 5\n"
        "Chapter 2 ........................ 12\n"
        "Chapter 3 ........................ 20\n\n"
        "Chapter 1\nBody text.\nChapter 2\nBody text.\nChapter 3\nBody text."
    )
    assert toc.detected and toc.usable and toc.shape == TocShape.FLAT_CHAPTER
    print("text normalization checks passed")


if __name__ == "__main__":
    main()
