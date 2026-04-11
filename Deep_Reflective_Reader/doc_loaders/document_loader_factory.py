from .text_document_loader import TextDocumentLoader
from .pdf_document_loader import PdfDocumentLoader


class DocumentLoaderFactory:
    def __init__(self):
        self.txt_loader = TextDocumentLoader()
        self.pdf_loader = PdfDocumentLoader()

    def get(self, doc_name: str):
        # 👉 最簡單判斷（可以之後優化）
        if doc_name.endswith(".pdf"):
            return self.pdf_loader
        return self.txt_loader