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
    print("toc projection checks passed")


if __name__ == "__main__":
    main()
