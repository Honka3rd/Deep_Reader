class RawTextRequiresOcrError(ValueError):
    """Raised when a document has no native text and requires OCR."""

    def __init__(self, doc_name: str, detail: str | None = None):
        self.doc_name = doc_name
        self.detail = detail
        message = doc_name if detail is None else f"{doc_name}:{detail}"
        super().__init__(message)


class RawTextOcrFailedError(ValueError):
    """Raised when OCR was enabled but did not produce usable text."""

    def __init__(self, doc_name: str, detail: str | None = None):
        self.doc_name = doc_name
        self.detail = detail
        message = doc_name if detail is None else f"{doc_name}:{detail}"
        super().__init__(message)
