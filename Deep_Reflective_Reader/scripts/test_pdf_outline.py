from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_outline import PdfOutlineEntry, PdfOutlineResult
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode


def main() -> None:
    loader = PdfDocumentLoader(base_dir="data/raw")
    result = loader.load_outline("许三观卖血记")
    assert result.present
    assert result.usable
    assert result.entries
    assert result.entries[0].title == "许三观卖血记"
    assert any(entry.title == "第一章" and entry.page_index == 4 for entry in result.entries)
    assert any(entry.level > 0 for entry in result.entries)
    assert "outline_validated" in result.reasons
    raw_text = loader.load("许三观卖血记")
    boundaries = loader.load_page_text_boundaries("许三观卖血记")
    document = StructuredDocumentBuilder().build(
        document_id="许三观卖血记",
        title="许三观卖血记",
        raw_text=raw_text,
        language=LanguageCode.ZH,
        outline=result,
        page_boundaries=boundaries,
    )
    assert document.parse_provenance["effective_parser_mode"] == "native_pdf_outline"
    assert document.parse_provenance["outline_projection"]["normalized_chapter_level"] == 1
    assert document.parse_provenance["outline_projection"]["root_wrapper_skipped"]
    assert len(document.chapters) >= 29
    assert document.chapters[0].title == "第一章"
    assert document.chapters[0].sections
    test_outline_level_zero_is_normalized_to_chapters()
    print("pdf outline checks passed")


def test_outline_level_zero_is_normalized_to_chapters() -> None:
    pages = [
        "目次\n目录页",
        "序論及全書設計\n序論正文",
        "第一篇 論勞働生產力改良的原因\n篇前說明",
        "第一章 分工論\n第一章正文",
        "第二章 分工的原由\n第二章正文",
        "第一節 基因於職業本身性質的不均等\n第一節正文",
    ]
    raw_text = "\n\n".join(pages)
    boundaries: list[PdfPageTextBoundary] = []
    cursor = 0
    for page_index, page_text in enumerate(pages):
        start = cursor
        end = start + len(page_text)
        boundaries.append(
            PdfPageTextBoundary(
                page_index=page_index,
                char_start=start,
                char_end=end,
                text=page_text,
            )
        )
        cursor = end + 2

    outline = PdfOutlineResult(
        present=True,
        usable=True,
        entries=[
            PdfOutlineEntry(title="目次", level=0, page_index=0, page_label="1"),
            PdfOutlineEntry(title="序論及全書設計", level=0, page_index=1, page_label="2"),
            PdfOutlineEntry(
                title="第一篇 論勞働生產力改良的原因",
                level=0,
                page_index=2,
                page_label="3",
            ),
            PdfOutlineEntry(title="第一章 分工論", level=1, page_index=3, page_label="4"),
            PdfOutlineEntry(title="第二章 分工的原由", level=1, page_index=4, page_label="5"),
            PdfOutlineEntry(
                title="第一節 基因於職業本身性質的不均等",
                level=2,
                page_index=5,
                page_label="6",
            ),
        ],
        reasons=["outline_validated"],
    )

    document = StructuredDocumentBuilder().build(
        document_id="國富論",
        title="國富論",
        raw_text=raw_text,
        language=LanguageCode.ZH,
        outline=outline,
        page_boundaries=boundaries,
    )

    assert document.parse_provenance["effective_parser_mode"] == "native_pdf_outline"
    assert document.parse_provenance["outline_projection"]["normalized_chapter_level"] == 0
    assert not document.parse_provenance["outline_projection"]["root_wrapper_skipped"]
    assert [chapter.title for chapter in document.chapters] == [
        "目次",
        "序論及全書設計",
        "第一篇 論勞働生產力改良的原因",
    ]
    assert [section.title for section in document.chapters[2].sections] == [
        "第一篇 論勞働生產力改良的原因",
        "第一章 分工論",
        "第二章 分工的原由",
        "第一節 基因於職業本身性質的不均等",
    ]
    assert "第一章正文" in document.chapters[2].sections[1].content
    assert "第二章正文" in document.chapters[2].sections[2].content


if __name__ == "__main__":
    main()
