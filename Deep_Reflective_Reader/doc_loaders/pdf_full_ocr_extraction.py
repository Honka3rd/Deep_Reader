"""Explicit full-page OCR; no native extraction or automatic decision policy."""

from pathlib import Path
import logging
import shutil
import subprocess
import tempfile

from .document_load_errors import RawTextOcrFailedError
from .pdf_execution_pass import PdfBudgetExceeded
from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason as Reason, PdfExtractionMode
from .pdf_extraction_result import PdfExtractionResult
from .pdf_page_evidence import PdfPageBoundaryEvidenceResult, PdfPageTextBoundary, PdfPageLayoutEvidence
from .ocr_review import enabled, region_record, review_record
from PIL import Image

logger = logging.getLogger(__name__)
CHINESE_QUALITY_REASONS = frozenset({
    "chinese_layout_coverage_uncertain", "chinese_column_label_uncertain",
    "chinese_vertical_region_quality_failed", "chinese_page_confidence_insufficient",
    "chinese_mixed_region_uncertain",
})


def extract_full_ocr(reader, *, loader, source, quality_version, doc_name, source_file_name,
                     source_sha256, outline_provider):
    with loader.preparation_scope():
        try:
            return _extract(reader, loader=loader, source=source, quality_version=quality_version, doc_name=doc_name,
                            source_file_name=source_file_name, source_sha256=source_sha256,
                            outline_provider=outline_provider)
        except PdfBudgetExceeded as error:
            raise PdfExtractionError(Reason.BUDGET_EXCEEDED, error.reason) from error
        except (RawTextOcrFailedError, OSError, subprocess.TimeoutExpired) as error:
            raise PdfExtractionError(Reason.EXTRACTION_FAILED, "Full-page OCR execution failed.") from error


def _extract(reader, *, loader, source, quality_version, doc_name, source_file_name, source_sha256, outline_provider):
    language = loader.ocr_language
    count = len(reader.pages)
    if loader.ocr_page_limit is not None and count > loader.ocr_page_limit:
        raise PdfExtractionError(Reason.BUDGET_EXCEEDED, "OCR page limit excludes required pages.")
    if not count:
        raise PdfExtractionError(Reason.NO_TEXT, "PDF has no pages.")
    if not shutil.which(loader.tesseract_cmd) or not shutil.which(loader.pdf_renderer_cmd):
        raise PdfExtractionError(Reason.ENGINE_UNAVAILABLE, "OCR engine or PDF renderer is unavailable.")
    languages = loader._run_process(
        [loader.tesseract_cmd, "--list-langs"], purpose="text_recovery", stage="metadata", page=-1,
    )
    installed = {line.strip() for line in languages.stdout.splitlines()}
    if languages.returncode or not set(loader.ocr_language.split("+")).issubset(installed):
        raise PdfExtractionError(Reason.ENGINE_UNAVAILABLE, "Requested OCR language packs are unavailable.")
    pages, quality, unresolved = [], [], []
    readable_area = uncertain_area = 0
    review_enabled = enabled()
    with tempfile.TemporaryDirectory(prefix="deep-reader-full-ocr-") as directory:
        root = Path(directory)
        snapshot = root / "source.pdf"
        snapshot.write_bytes(source)
        for index in range(count):
            images = loader._render_page_images(file_path=snapshot, page_index=index, output_dir=root)
            if len(images) != 1:
                raise PdfExtractionError(Reason.EXTRACTION_FAILED, f"Page {index + 1}: expected one full-page render.")
            # Only a rendered blank page may skip recognition, never a native-text decision.
            if loader._render_is_uniform(images[0]):
                pages.append("")
                quality.append({"page_index": index, "blank_render_confirmed": True})
                images[0].unlink(missing_ok=True)
                continue
            try:
                text, candidates = loader._ocr_image_text_with_quality(
                    image_path=images[0], doc_name=doc_name, page_index=index,
                    **({"allow_review":True} if review_enabled else {}),
                )
            except RawTextOcrFailedError as error:
                # Only stable internal codes are public; never return subprocess/source text.
                parts = (error.detail or "").split(":")
                reason = parts[0]
                region = next((p for p in parts[1:] if p.startswith("region=")
                               and p[7:].isdigit() and len(p) < 16), None)
                logger.warning("full_ocr_page_failed page=%s reason=%s region=%s",
                               index+1, reason if reason in CHINESE_QUALITY_REASONS else "execution_failed", region)
                if reason in CHINESE_QUALITY_REASONS:
                    if review_enabled:
                        text, candidates = "", []
                    else:
                        detail = f"OCR quality rejected page {index+1}: {reason}"
                        if region:
                            detail += f" ({region})."
                        raise PdfExtractionError(Reason.QUALITY_FAILED, detail) from error
                else:
                    raise PdfExtractionError(Reason.EXTRACTION_FAILED,
                                             f"OCR execution failed on page {index+1}.") from error
            summary = loader._summarize_page_ocr_quality(page_index=index, candidates=candidates)
            if not text.strip() or not summary["selected_passed"]:
                if not review_enabled:
                    raise PdfExtractionError(Reason.QUALITY_FAILED, f"OCR quality rejected page {index + 1}.")
                with Image.open(images[0]) as rendered:
                    bbox = [0,0,*rendered.size]
                uncertain_area += bbox[2]*bbox[3]
                unresolved.append(region_record(index,0,bbox,loader.pdf_render_dpi,
                    "page_quality_rejected", candidates[0].text if candidates else ""))
                text = ""
            elif review_enabled:
                regions = summary.get("regions",[])
                if regions:
                    for i, region in enumerate(regions):
                        x0,y0,x1,y1 = region["bbox"]
                        area = (x1-x0)*(y1-y0)
                        if region.get("accepted") is False:
                            uncertain_area += area
                            unresolved.append(region_record(index,i,region["bbox"],loader.pdf_render_dpi,
                                "recognition_uncertain",region.get("raw_candidate") or ""))
                        else:
                            readable_area += area
                else:
                    with Image.open(images[0]) as rendered:
                        readable_area += rendered.width*rendered.height
            pages.append(text.strip())
            quality.append(summary)
            images[0].unlink(missing_ok=True)
    if not any(pages):
        raise PdfExtractionError(Reason.QUALITY_FAILED if unresolved else Reason.NO_TEXT,
                                 "No readable OCR text." if unresolved else "All rendered PDF pages are blank.")
    if unresolved and (sum(len(p.strip()) for p in pages) < 40 or readable_area < uncertain_area):
        raise PdfExtractionError(Reason.QUALITY_FAILED,"Insufficient readable coverage for reviewable OCR import.")
    try:
        labels = reader.page_labels or []
    except Exception:
        labels = []
    boundaries, cursor, seen = [], 0, False
    for index, text in enumerate(pages):
        if text and seen:
            cursor += 1
        start = cursor
        cursor += len(text)
        boundaries.append(PdfPageTextBoundary(
            page_index=index, char_start=start, char_end=cursor, text=text,
            page_label=labels[index] if index < len(labels) else None, source_sha256=source_sha256,
        ))
        seen = seen or bool(text)

    def layout():
        # Text inventory only; do not re-enter the auto coordinate-OCR path.
        return [PdfPageLayoutEvidence(
            page_index=index, width=None, height=None, native_text_chars=0, image_count=0,
            source_file_name=source_file_name, source_sha256=source_sha256,
            ocr_text=text, ocr_engine="tesseract", ocr_language=language,
            evidence=["full_page_ocr_text_inventory", "image_count_unknown", "coordinates_unavailable"],
        ) for index, text in enumerate(pages)]

    return PdfExtractionResult(
        text="\n".join(text for text in pages if text), pages=tuple(pages),
        boundary_evidence=PdfPageBoundaryEvidenceResult(
            doc_name, source_file_name, source_sha256, count, boundaries,
        ), effective_mode=PdfExtractionMode.OCR, extraction_version="full-page-ocr-v1",
        empty_page_indexes=tuple(i for i, text in enumerate(pages) if not text),
        warnings=("ocr_transcription_accuracy_unverified",) + (("source_needs_review",) if unresolved else ()),
        layout_provider=layout, outline_provider=outline_provider,
        provenance={"source_sha256": source_sha256, "mode": "ocr", "page_count": count,
                    "ocr_language": language, "render_dpi": loader.pdf_render_dpi,
                    "quality_version": quality_version,
                    "source_review":review_record(source_sha256,unresolved)},
        page_quality=tuple(quality),
    )
