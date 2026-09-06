from doc_loaders.pdf_document_loader import PdfDocumentLoader
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
    assert len(document.chapters) >= 29
    assert document.chapters[1].title == "第一章"
    assert document.chapters[1].sections
    print("pdf outline checks passed")


if __name__ == "__main__":
    main()
