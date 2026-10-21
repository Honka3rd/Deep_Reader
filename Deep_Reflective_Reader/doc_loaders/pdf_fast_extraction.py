"""Native text only: no OCR policy, graphics inspection or subprocess dependency."""

from collections.abc import Callable
import logging
import unicodedata

from pypdf import PdfReader

from logging_config import log_stage
from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason, PdfExtractionMode
from .pdf_extraction_result import PdfExtractionResult
from .pdf_outline import PdfOutlineResult
from .pdf_page_evidence import (
    PdfPageBoundaryEvidenceResult, PdfPageLayoutEvidence, PdfPageTextBoundary,
)


logger = logging.getLogger(__name__)


def extract_fast(
    reader: PdfReader, *, doc_name: str, source_file_name: str, source_sha256: str,
    outline_provider: Callable[[], PdfOutlineResult],
) -> PdfExtractionResult:
    pages = []
    warnings = ["native_only_visual_coverage_unverified"]
    with log_stage(logger, "pdf_fast_native_extraction"):
        for index, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as error:
                raise PdfExtractionError(
                    PdfExtractionFailureReason.EXTRACTION_FAILED,
                    f"Native extraction failed on page {index + 1}; try OCR explicitly.",
                ) from error
            # Diagnose only text encoding; never infer language or inspect graphics.
            invalid = sum(
                char == "\ufffd" or (unicodedata.category(char) == "Cc" and not char.isspace())
                for char in text
            )
            if invalid:
                warnings.append(f"native_text_encoding_suspect:page={index + 1}:chars={invalid}")
                if all(char.isspace() or char == "\ufffd" or unicodedata.category(char) == "Cc" for char in text):
                    raise PdfExtractionError(
                        PdfExtractionFailureReason.QUALITY_FAILED,
                        f"Native text is unusable on page {index + 1}; try OCR explicitly.",
                    )
            pages.append(text)
    if not any(pages):
        raise PdfExtractionError(
            PdfExtractionFailureReason.NO_TEXT,
            "No native PDF text was found; select OCR or auto explicitly if this is a scan.",
        )
    empty = tuple(index for index, text in enumerate(pages) if not text)
    warnings.extend(f"native_text_empty:page={index + 1}" for index in empty)
    try:
        labels = reader.page_labels or []
    except Exception:
        labels = []
        warnings.append("page_labels_unavailable")
    boundaries = []
    cursor = 0
    seen_text = False
    for index, text in enumerate(pages):
        if text and seen_text:
            cursor += 1
        start = cursor
        cursor += len(text)
        boundaries.append(PdfPageTextBoundary(
            page_index=index, char_start=start, char_end=cursor, text=text,
            page_label=labels[index] if index < len(labels) else None,
            source_sha256=source_sha256,
        ))
        seen_text = seen_text or bool(text)

    # Capture this source snapshot, never re-enter the loader's auto layout path.
    def native_layout() -> list[PdfPageLayoutEvidence]:
        evidence = []
        for index, page in enumerate(reader.pages):
            flags = ["fast_native_only", "image_count_unknown", "visual_coverage_unverified"]
            try:
                width, height = int(page.mediabox.width), int(page.mediabox.height)
                rotation = int(page.rotation or 0) % 360
            except Exception:
                width = height = rotation = None
                flags.append("page_geometry_unavailable")
            evidence.append(PdfPageLayoutEvidence(
                page_index=index, width=width, height=height,
                native_text_chars=len(pages[index]), image_count=0,
                source_file_name=source_file_name, source_sha256=source_sha256,
                rotation_degrees=rotation, ocr_text=pages[index], evidence=flags,
            ))
        return evidence

    logger.info("pdf_fast_complete pages=%s native_chars=%s empty_pages=%s",
                len(pages), sum(map(len, pages)), len(empty))
    return PdfExtractionResult(
        text="\n".join(text for text in pages if text), pages=tuple(pages),
        boundary_evidence=PdfPageBoundaryEvidenceResult(
            doc_name=doc_name, source_file_name=source_file_name, source_sha256=source_sha256,
            page_count=len(pages), boundaries=boundaries,
        ),
        effective_mode=PdfExtractionMode.FAST, extraction_version="fast-native-v1",
        empty_page_indexes=empty, warnings=tuple(warnings),
        layout_provider=native_layout, outline_provider=outline_provider,
    )
