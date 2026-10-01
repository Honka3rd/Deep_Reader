from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace

from pypdf import PdfReader

from .abstract_document_loader import AbstractDocumentLoader
from .document_load_errors import (
    RawTextOcrFailedError,
    RawTextOcrLowQualityError,
    RawTextRequiresOcrError,
)
from .pdf_ocr_language_policy import normalize_tesseract_language_config
from .pdf_outline import PdfOutlineEntry, PdfOutlineResult
from .pdf_page_evidence import (
    PdfPageBoundaryEvidenceResult,
    PdfPageLayoutEvidence,
    PdfPageTextBoundary,
    analyze_ocr_tsv,
)
from .raw_data_paths import resolve_raw_data_dir


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfInspectionMetrics:
    """Lightweight PDF text/image inspection result."""

    page_count: int
    native_text_chars: int
    pages_with_images: int
    pages_with_fonts: int

    @property
    def image_page_ratio(self) -> float:
        if self.page_count <= 0:
            return 0.0
        return self.pages_with_images / self.page_count

    @property
    def font_page_ratio(self) -> float:
        if self.page_count <= 0:
            return 0.0
        return self.pages_with_fonts / self.page_count

    @property
    def is_scanned_image_pdf(self) -> bool:
        return (
            self.page_count > 0
            and self.native_text_chars == 0
            and self.image_page_ratio >= 0.8
            and self.font_page_ratio <= 0.1
        )

    @property
    def is_predominantly_scanned_image_pdf(self) -> bool:
        """Treat sparse accidental OCR/text fragments as a scanned PDF signal."""
        return (
            self.page_count > 0
            and self.native_text_chars <= max(100, self.page_count * 2)
            and self.image_page_ratio >= 0.8
            and self.font_page_ratio <= 0.1
        )


@dataclass(frozen=True)
class OcrCandidateQuality:
    """Deterministic quality summary for one OCR candidate."""

    psm: str
    text: str
    score: float
    passed: bool
    reasons: list[str]
    cjk_ratio: float
    latin_ratio: float
    symbol_ratio: float
    single_token_ratio: float


class PdfDocumentLoader(AbstractDocumentLoader):
    """Load PDF document files and join extracted page text."""
    base_dir: Path

    def __init__(
        self,
        base_dir: str | Path | None = None,
        ocr_enabled: bool | None = None,
        ocr_language: str | None = None,
        tesseract_cmd: str | None = None,
        ocr_page_limit: int | None = None,
        ocr_timeout_seconds: int = 60,
        pdf_renderer_cmd: str | None = None,
        pdf_render_dpi: int = 200,
    ):
        """Initialize object state and injected dependencies.

Args:
    base_dir: Base dir.
"""
        self.base_dir = resolve_raw_data_dir(base_dir)
        self.ocr_enabled = (
            self._env_flag("DEEP_READER_PDF_OCR_ENABLED")
            if ocr_enabled is None
            else ocr_enabled
        )
        self.ocr_language = (
            normalize_tesseract_language_config(ocr_language)
            if ocr_language is not None
            else normalize_tesseract_language_config(
                os.environ.get("DEEP_READER_PDF_OCR_LANGUAGE")
            )
        )
        self.tesseract_cmd = (
            tesseract_cmd
            or os.environ.get("DEEP_READER_TESSERACT_CMD", "tesseract").strip()
            or "tesseract"
        )
        self.ocr_page_limit = ocr_page_limit
        self.ocr_timeout_seconds = ocr_timeout_seconds
        self.pdf_renderer_cmd = (
            pdf_renderer_cmd
            or os.environ.get("DEEP_READER_PDF_RENDERER_CMD", "pdftoppm").strip()
            or "pdftoppm"
        )
        self.pdf_render_dpi = int(
            os.environ.get("DEEP_READER_PDF_RENDER_DPI", str(pdf_render_dpi))
        )
        self.last_ocr_provenance: dict[str, object] | None = None
        self.last_ocr_pages: list[str] | None = None
        self.last_ocr_quality: list[dict[str, object]] = []

    def load(self, doc_name: str) -> str:
        """Load persisted artifact and return parsed object/data.

Args:
    doc_name: Logical document name; supports both ``name`` and ``name.pdf``.

Returns:
    Concatenated text extracted from all PDF pages."""
        file_path = self._resolve_file_path(doc_name)

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = PdfReader(str(file_path))
        text, metrics = self._extract_text_and_metrics(reader)
        logger.info(
            "pdf_inspected doc=%s file=%s pages=%s native_text_chars=%s image_pages=%s font_pages=%s scanned=%s",
            doc_name,
            file_path.name,
            metrics.page_count,
            metrics.native_text_chars,
            metrics.pages_with_images,
            metrics.pages_with_fonts,
            metrics.is_predominantly_scanned_image_pdf,
        )
        if metrics.is_predominantly_scanned_image_pdf:
            if self.ocr_enabled:
                logger.info(
                    "ocr_started doc=%s language=%s page_limit=%s persistence=memory_then_ocr_runs",
                    doc_name,
                    self.ocr_language,
                    self.ocr_page_limit or len(reader.pages),
                )
                return self._load_text_with_ocr(
                    doc_name=doc_name,
                    file_path=file_path,
                    reader=reader,
                )
            if not text:
                raise RawTextRequiresOcrError(
                    doc_name=doc_name,
                    detail=(
                        f"pages={metrics.page_count},"
                        f"image_pages={metrics.pages_with_images},"
                        f"font_pages={metrics.pages_with_fonts}"
                    ),
                )
        return text

    def load_pages(self, doc_name: str) -> list[str]:
        """Load canonical text while preserving one text value per PDF page."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = PdfReader(str(file_path))
        native_pages = [((page.extract_text() or "").strip()) for page in reader.pages]
        if any(native_pages):
            _, metrics = self._extract_text_and_metrics(reader)
            if not metrics.is_predominantly_scanned_image_pdf:
                return native_pages
        else:
            _, metrics = self._extract_text_and_metrics(reader)
        if not metrics.is_predominantly_scanned_image_pdf:
            return native_pages
        if not self.ocr_enabled:
            raise RawTextRequiresOcrError(doc_name=doc_name, detail="page_aware_load")
        return self._load_pages_with_ocr(
            doc_name=doc_name,
            file_path=file_path,
            reader=reader,
        )

    def load_page_layout_evidence(
        self,
        doc_name: str,
        *,
        candidate_page_limit: int | None = None,
    ) -> list[PdfPageLayoutEvidence]:
        """Collect page inventory and coordinate OCR evidence for candidate pages."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        reader = PdfReader(str(file_path))
        source_sha256 = self._file_sha256(file_path)
        evidence: list[PdfPageLayoutEvidence] = []
        page_limit = candidate_page_limit or len(reader.pages)
        for page_index, page in enumerate(reader.pages):
            native_text = (page.extract_text() or "").strip()
            try:
                images = list(getattr(page, "images", []))
            except Exception:
                images = []
            width, height = self._page_dimensions(page, images)
            if page_index >= page_limit or not self.ocr_enabled:
                evidence.append(
                    PdfPageLayoutEvidence(
                        page_index=page_index,
                        width=width,
                        height=height,
                        native_text_chars=len(native_text),
                        image_count=len(images),
                        source_file_name=file_path.name,
                        source_sha256=source_sha256,
                        ocr_text=native_text,
                        analysis_stage="inventory",
                        analysis_cost_tier="cheap_inventory",
                        ocr_pass_count=0,
                        cache_key=self._layout_cache_key(
                            source_sha256=source_sha256,
                            page_index=page_index,
                            stage="inventory",
                        ),
                        cache_hit=False,
                        high_cost_analysis_permitted=False,
                        evidence=["page_inventory"],
                    )
                )
                continue
            with tempfile.TemporaryDirectory(prefix="deep-reader-page-layout-render-") as temp_dir:
                try:
                    image_paths = self._render_page_images(
                        file_path=file_path,
                        page_index=page_index,
                        output_dir=Path(temp_dir),
                    )
                except RawTextOcrFailedError:
                    image_paths = []
                evidence.append(
                    self._build_page_layout_evidence(
                        page_index=page_index,
                        width=width,
                        height=height,
                        native_text_chars=len(native_text),
                        images=images,
                        image_paths=image_paths,
                        source_file_name=file_path.name,
                        source_sha256=source_sha256,
                    )
                )
        return evidence

    def load_page_text_boundaries(self, doc_name: str) -> list[PdfPageTextBoundary]:
        """Return canonical page-to-character spans using the page-aware loader."""
        evidence = self.load_page_boundary_evidence(doc_name)
        return list(evidence.boundaries)

    def load_page_boundary_evidence(self, doc_name: str) -> PdfPageBoundaryEvidenceResult:
        """Return compact page boundary evidence for manual TOC anchors."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        source_sha256 = self._file_sha256(file_path)
        reader = PdfReader(str(file_path))
        pages = self.load_pages(doc_name)
        separator = "\n\n"
        native_pages: list[str] = []
        page_labels = self._reader_page_labels(reader)
        try:
            native_pages = [page.extract_text() or "" for page in reader.pages]
            if any(native_pages):
                separator = "\n"
        except Exception:
            pass
        boundaries: list[PdfPageTextBoundary] = []
        cursor = 0
        has_native_text = bool(native_pages and any(native_pages))
        seen_native_page = False
        for page_index, text in enumerate(pages):
            if has_native_text and not native_pages[page_index]:
                boundaries.append(
                    PdfPageTextBoundary(
                        page_index=page_index,
                        char_start=cursor,
                        char_end=cursor,
                        text="",
                        page_label=(
                            page_labels[page_index]
                            if page_index < len(page_labels)
                            else None
                        ),
                        source_sha256=source_sha256,
                    )
                )
                continue
            if has_native_text:
                text = native_pages[page_index]
            if has_native_text and seen_native_page:
                cursor += len(separator)
            start = cursor
            cursor += len(text)
            boundaries.append(
                PdfPageTextBoundary(
                    page_index=page_index,
                    char_start=start,
                    char_end=cursor,
                    text=text,
                    page_label=(
                        page_labels[page_index]
                        if page_index < len(page_labels)
                        else None
                    ),
                    source_sha256=source_sha256,
                )
            )
            seen_native_page = seen_native_page or has_native_text
            if not has_native_text and page_index + 1 < len(pages):
                cursor += len(separator)
        return PdfPageBoundaryEvidenceResult(
            doc_name=doc_name,
            source_file_name=file_path.name,
            source_sha256=source_sha256,
            page_count=len(reader.pages),
            boundaries=boundaries,
        )

    def load_outline(self, doc_name: str) -> PdfOutlineResult:
        """Read and validate native PDF bookmarks without invoking OCR."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        reader = PdfReader(str(file_path))
        source_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if "/Outlines" not in reader.trailer["/Root"]:
            return PdfOutlineResult(
                present=False,
                usable=False,
                reasons=["outline_missing"],
                source_sha256=source_sha256,
            )

        page_labels = reader.page_labels or []
        entries: list[PdfOutlineEntry] = []
        reasons: list[str] = []

        def visit(items: list[object], level: int) -> None:
            for item in items:
                if isinstance(item, list):
                    visit(item, level + 1)
                    continue
                title = str(getattr(item, "title", "") or "").strip()
                if not title:
                    reasons.append("outline_entry_missing_title")
                    continue
                try:
                    page_index = reader.get_destination_page_number(item)
                except Exception:
                    page_index = None
                if page_index is not None and not 0 <= page_index < len(reader.pages):
                    page_index = None
                entries.append(
                    PdfOutlineEntry(
                        title=title,
                        level=max(0, level),
                        page_index=page_index,
                        page_label=(
                            page_labels[page_index]
                            if page_index is not None and page_index < len(page_labels)
                            else None
                        ),
                        destination_type=(
                            None if getattr(item, "typ", None) is None else str(item.typ)
                        ),
                    )
                )

        try:
            visit(reader.outline, 0)
        except Exception as error:
            return PdfOutlineResult(
                present=True,
                usable=False,
                reasons=["outline_parse_failed", type(error).__name__],
                source_sha256=source_sha256,
            )

        if not entries:
            reasons.append("outline_empty")
        if any(entry.page_index is None for entry in entries):
            reasons.append("outline_missing_destinations")
        if any(
            right.page_index is not None
            and left.page_index is not None
            and right.page_index < left.page_index
            for left, right in zip(entries, entries[1:])
        ):
            reasons.append("outline_page_order_not_monotonic")
        usable = bool(entries) and all(entry.page_index is not None for entry in entries)
        if usable:
            reasons.append("outline_validated")
        return PdfOutlineResult(
            present=True,
            usable=usable,
            entries=entries,
            reasons=reasons,
            source_sha256=source_sha256,
        )

    def _build_page_layout_evidence(
        self,
        *,
        page_index: int,
        width: int | None,
        height: int | None,
        native_text_chars: int,
        images: list[object],
        image_paths: list[Path] | None = None,
        source_file_name: str | None = None,
        source_sha256: str | None = None,
    ) -> PdfPageLayoutEvidence:
        """Run coordinate OCR on one candidate page without changing raw text."""
        if shutil.which(self.tesseract_cmd) is None:
            return PdfPageLayoutEvidence(
                page_index=page_index,
                width=width,
                height=height,
                native_text_chars=native_text_chars,
                image_count=len(images),
                source_file_name=source_file_name,
                source_sha256=source_sha256,
                analysis_stage="inventory",
                analysis_cost_tier="cheap_inventory",
                ocr_pass_count=0,
                cache_key=self._layout_cache_key(
                    source_sha256=source_sha256,
                    page_index=page_index,
                    stage="inventory",
                ),
                cache_hit=False,
                failure_reason="tesseract_unavailable",
                high_cost_analysis_permitted=False,
                evidence=["tesseract_unavailable"],
            )
        with tempfile.TemporaryDirectory(prefix="deep-reader-pdf-layout-") as temp_dir:
            candidates: list[tuple[float, PdfPageLayoutEvidence]] = []
            sources = image_paths or []
            if not sources:
                sources = []
                for image_index, image in enumerate(images):
                    suffix = Path(getattr(image, "name", "")).suffix or ".png"
                    image_path = Path(temp_dir) / f"page-{page_index + 1}-{image_index + 1}{suffix}"
                    image_path.write_bytes(image.data)
                    sources.append(image_path)
            for psm in ("11", "5"):
                parts: list[str] = []
                for image_path in sources:
                    result = subprocess.run(
                        [
                            self.tesseract_cmd,
                            str(image_path),
                            "stdout",
                            "--psm",
                            psm,
                            "-l",
                            self.ocr_language,
                            "tsv",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.ocr_timeout_seconds,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        parts.append(result.stdout)
                candidate = analyze_ocr_tsv(
                    page_index=page_index,
                    width=width,
                    height=height,
                    native_text_chars=native_text_chars,
                    image_count=max(len(images), len(sources)),
                    tsv_text="\n".join(parts),
                    source_file_name=source_file_name,
                    source_sha256=source_sha256,
                    ocr_language=self.ocr_language,
                    render_dpi=self.pdf_render_dpi,
                    ocr_pass_count=len(sources),
                    cache_key=self._layout_cache_key(
                        source_sha256=source_sha256,
                        page_index=page_index,
                        stage="coordinate_ocr",
                        psm=psm,
                    ),
                    cache_hit=False,
                    high_cost_analysis_permitted=False,
                )
                candidate = replace(candidate, evidence=[*candidate.evidence, f"tesseract_psm_{psm}"])
                quality = (
                    candidate.ocr_confidence
                    + min(candidate.word_count / 100.0, 0.3)
                    + (0.45 if candidate.writing_mode == "vertical" else 0.0)
                    + (
                        0.25
                        if candidate.writing_mode == "vertical"
                        and candidate.line_count <= max(candidate.column_count * 6, 20)
                        else 0.0
                    )
                    + (0.05 if candidate.column_count >= 3 else 0.0)
                )
                candidates.append((quality, candidate))
            if candidates:
                return max(candidates, key=lambda item: item[0])[1]
            return analyze_ocr_tsv(
                page_index=page_index,
                width=width,
                height=height,
                native_text_chars=native_text_chars,
                image_count=max(len(images), len(sources)),
                tsv_text="",
                source_file_name=source_file_name,
                source_sha256=source_sha256,
                ocr_language=self.ocr_language,
                render_dpi=self.pdf_render_dpi,
                ocr_pass_count=0,
                cache_key=self._layout_cache_key(
                    source_sha256=source_sha256,
                    page_index=page_index,
                    stage="coordinate_ocr",
                ),
                cache_hit=False,
                failure_reason="no_ocr_sources",
                high_cost_analysis_permitted=False,
            )

    @staticmethod
    def _layout_cache_key(
        *,
        source_sha256: str | None,
        page_index: int,
        stage: str,
        psm: str | None = None,
    ) -> str | None:
        if not source_sha256:
            return None
        psm_part = f":psm:{psm}" if psm is not None else ""
        return f"pdf-layout:v1:{source_sha256}:page:{page_index}:stage:{stage}{psm_part}"

    @staticmethod
    def _page_dimensions(page: object, images: list[object]) -> tuple[int | None, int | None]:
        if images:
            try:
                return tuple(int(value) for value in images[0].image.size)  # type: ignore[return-value]
            except Exception:
                pass
        try:
            box = page.mediabox
            return int(float(box.width)), int(float(box.height))
        except Exception:
            return None, None

    def _load_text_with_ocr(
        self,
        doc_name: str,
        file_path: Path,
        reader: PdfReader,
    ) -> str:
        if shutil.which(self.tesseract_cmd) is None:
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_not_found:{self.tesseract_cmd}",
            )

        provenance = self._build_ocr_provenance(
            doc_name=doc_name,
            file_path=file_path,
            page_count=len(reader.pages),
        )
        pages = self._load_pages_with_ocr(
            doc_name=doc_name,
            file_path=file_path,
            reader=reader,
        )
        ocr_text = "\n\n".join(pages)
        if not ocr_text.strip():
            logger.error("ocr_failed doc=%s detail=empty_ocr_text", doc_name)
            raise RawTextOcrFailedError(doc_name=doc_name, detail="empty_ocr_text")
        logger.info(
            "ocr_completed doc=%s text_chars=%s persistence=ocr_runs",
            doc_name,
            len(ocr_text),
        )
        return ocr_text

    def _load_pages_with_ocr(
        self,
        *,
        doc_name: str,
        file_path: Path,
        reader: PdfReader,
    ) -> list[str]:
        """OCR one page at a time and retain page-preserving output in memory."""
        if shutil.which(self.tesseract_cmd) is None:
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_not_found:{self.tesseract_cmd}",
            )

        provenance = self._build_ocr_provenance(
            doc_name=doc_name,
            file_path=file_path,
            page_count=len(reader.pages),
            schema_version=2,
        )
        if self.last_ocr_provenance == provenance and self.last_ocr_pages is not None:
            logger.info(
                "ocr_pages_memory_hit doc=%s pages=%s",
                doc_name,
                len(self.last_ocr_pages),
            )
            return list(self.last_ocr_pages)

        pages: list[str] = []
        quality_by_page: list[dict[str, object]] = []
        page_limit = self.ocr_page_limit or len(reader.pages)
        with tempfile.TemporaryDirectory(prefix="deep-reader-pdf-ocr-pages-") as temp_dir:
            temp_path = Path(temp_dir)
            for page_index, page in enumerate(reader.pages[:page_limit]):
                recognized_parts: list[str] = []
                page_candidates: list[OcrCandidateQuality] = []
                image_paths = self._render_page_images(
                    file_path=file_path,
                    page_index=page_index,
                    output_dir=temp_path,
                )
                for image_path in image_paths:
                    recognized_text, candidates = self._ocr_image_text_with_quality(
                        image_path=image_path,
                        doc_name=doc_name,
                        page_index=page_index,
                    )
                    page_candidates.extend(candidates)
                    if recognized_text:
                        recognized_parts.append(recognized_text)
                pages.append("\n\n".join(recognized_parts))
                quality_by_page.append(
                    self._summarize_page_ocr_quality(
                        page_index=page_index,
                        candidates=page_candidates,
                    )
                )
                if (page_index + 1) % 10 == 0 or page_index + 1 == page_limit:
                    logger.info(
                        "ocr_progress doc=%s page=%s/%s recognized_chars=%s",
                        doc_name,
                        page_index + 1,
                        page_limit,
                        sum(len(page_text) for page_text in pages),
                    )

        if not any(page.strip() for page in pages):
            if self._has_nonempty_ocr_candidate(quality_by_page):
                self._raise_if_ocr_quality_unusable(
                    doc_name=doc_name,
                    quality_by_page=quality_by_page,
                )
            raise RawTextOcrFailedError(doc_name=doc_name, detail="empty_ocr_pages")
        self._raise_if_ocr_quality_unusable(
            doc_name=doc_name,
            quality_by_page=quality_by_page,
        )
        self.last_ocr_provenance = provenance
        self.last_ocr_pages = list(pages)
        self.last_ocr_quality = list(quality_by_page)
        logger.info(
            "ocr_pages_completed doc=%s pages=%s text_chars=%s persistence=memory_then_ocr_runs",
            doc_name,
            len(pages),
            sum(len(page) for page in pages),
        )
        return pages

    def _ocr_image_text(self, *, image_path: Path, doc_name: str, page_index: int) -> str:
        """Return the best OCR candidate text for one rendered page image."""
        text, _ = self._ocr_image_text_with_quality(
            image_path=image_path,
            doc_name=doc_name,
            page_index=page_index,
        )
        return text

    def _ocr_image_text_with_quality(
        self,
        *,
        image_path: Path,
        doc_name: str,
        page_index: int,
    ) -> tuple[str, list[OcrCandidateQuality]]:
        """Compare bounded Tesseract segmentation candidates deterministically."""
        candidates: list[OcrCandidateQuality] = []
        for psm in (None, "5", "6", "11"):
            command = [self.tesseract_cmd, str(image_path), "stdout", "-l", self.ocr_language]
            if psm is not None:
                command.extend(["--psm", psm])
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.ocr_timeout_seconds,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or f"exit_code={result.returncode}"
                raise RawTextOcrFailedError(doc_name=doc_name, detail=detail)
            text = result.stdout.strip()
            candidates.append(
                self._score_ocr_candidate(
                    psm="default" if psm is None else psm,
                    text=text,
                )
            )
        selected = max(candidates, key=lambda candidate: candidate.score, default=None)
        selected_text = selected.text if selected is not None and selected.passed else ""
        logger.info(
            "ocr_candidate_selected doc=%s page=%s psm=%s score=%s passed=%s reasons=%s",
            doc_name,
            page_index + 1,
            selected.psm if selected is not None else "none",
            round(selected.score, 4) if selected is not None else 0.0,
            selected.passed if selected is not None else False,
            ",".join(selected.reasons) if selected is not None else "no_candidates",
        )
        return selected_text, candidates

    def _score_ocr_candidate(self, *, psm: str, text: str) -> OcrCandidateQuality:
        normalized = text.strip()
        cjk_count = len(re.findall(r"[\u3400-\u9fff]", normalized))
        latin_count = len(re.findall(r"[A-Za-z]", normalized))
        digit_count = len(re.findall(r"[0-9]", normalized))
        signal_count = cjk_count + latin_count + digit_count
        non_space_count = len(re.findall(r"\S", normalized))
        symbol_count = max(0, non_space_count - signal_count)
        tokens = re.findall(r"\S+", normalized)
        single_tokens = [token for token in tokens if len(token) <= 2]
        cjk_ratio = cjk_count / signal_count if signal_count else 0.0
        latin_ratio = latin_count / signal_count if signal_count else 0.0
        symbol_ratio = symbol_count / non_space_count if non_space_count else 1.0
        single_token_ratio = len(single_tokens) / len(tokens) if tokens else 1.0
        multilingual_or_chinese = "chi" in self.ocr_language.casefold()

        if multilingual_or_chinese:
            score = (
                (0.55 * cjk_ratio)
                + min(cjk_count / 120.0, 0.25)
                + min(non_space_count / 500.0, 0.1)
                - (0.35 * latin_ratio)
                - (0.25 * symbol_ratio)
                - (0.3 * single_token_ratio)
            )
            threshold = 0.28
        else:
            score = (
                (0.45 * min((latin_count + digit_count) / 80.0, 1.0))
                + (0.25 * (1.0 - symbol_ratio))
                + min(non_space_count / 500.0, 0.15)
                - (0.25 * single_token_ratio)
            )
            threshold = 0.22

        reasons: list[str] = []
        if not normalized:
            reasons.append("empty_text")
        if multilingual_or_chinese and cjk_ratio < 0.35:
            reasons.append("low_cjk_ratio")
        if multilingual_or_chinese and latin_ratio > 0.45:
            reasons.append("latin_heavy")
        if symbol_ratio > 0.35:
            reasons.append("symbol_heavy")
        if single_token_ratio > 0.82 and len(tokens) >= 20:
            reasons.append("fragmented_tokens")
        if non_space_count < 20:
            reasons.append("too_short")
        passed = bool(normalized) and score >= threshold
        if passed:
            reasons.append("quality_passed")
        else:
            reasons.append("quality_rejected")
        return OcrCandidateQuality(
            psm=psm,
            text=normalized,
            score=round(score, 4),
            passed=passed,
            reasons=reasons,
            cjk_ratio=round(cjk_ratio, 4),
            latin_ratio=round(latin_ratio, 4),
            symbol_ratio=round(symbol_ratio, 4),
            single_token_ratio=round(single_token_ratio, 4),
        )

    def _summarize_page_ocr_quality(
        self,
        *,
        page_index: int,
        candidates: list[OcrCandidateQuality],
    ) -> dict[str, object]:
        selected = max(candidates, key=lambda candidate: candidate.score, default=None)
        return {
            "page_index": page_index,
            "selected_psm": selected.psm if selected is not None else None,
            "selected_score": selected.score if selected is not None else 0.0,
            "selected_passed": selected.passed if selected is not None else False,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "psm": candidate.psm,
                    "score": candidate.score,
                    "passed": candidate.passed,
                    "reasons": list(candidate.reasons),
                    "cjk_ratio": candidate.cjk_ratio,
                    "latin_ratio": candidate.latin_ratio,
                    "symbol_ratio": candidate.symbol_ratio,
                    "single_token_ratio": candidate.single_token_ratio,
                    "text_chars": len(candidate.text),
                }
                for candidate in candidates
            ],
        }

    def _raise_if_ocr_quality_unusable(
        self,
        *,
        doc_name: str,
        quality_by_page: list[dict[str, object]],
    ) -> None:
        if not quality_by_page:
            raise RawTextOcrLowQualityError(doc_name=doc_name, detail="no_quality_evidence")
        passed_pages = [
            page
            for page in quality_by_page
            if bool(page.get("selected_passed"))
        ]
        required_passed_pages = max(1, int((len(quality_by_page) * 0.35) + 0.999))
        if len(passed_pages) >= required_passed_pages:
            return
        best_page = max(
            quality_by_page,
            key=lambda page: float(page.get("selected_score", 0.0)),
        )
        detail = (
            "document_quality_gate_failed:"
            f"passed_pages={len(passed_pages)}/{len(quality_by_page)},"
            f"best_psm={best_page.get('selected_psm')},"
            f"best_score={best_page.get('selected_score')}"
        )
        logger.warning("ocr_low_quality doc=%s detail=%s", doc_name, detail)
        self.last_ocr_quality = list(quality_by_page)
        raise RawTextOcrLowQualityError(doc_name=doc_name, detail=detail)

    def _has_nonempty_ocr_candidate(self, quality_by_page: list[dict[str, object]]) -> bool:
        for page in quality_by_page:
            candidates = page.get("candidates")
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if isinstance(candidate, dict) and int(candidate.get("text_chars", 0)) > 0:
                    return True
        return False

    def _render_page_images(
        self,
        *,
        file_path: Path,
        page_index: int,
        output_dir: Path,
    ) -> list[Path]:
        """Render one complete PDF page so OCR does not depend on image XObject decoding."""
        if shutil.which(self.pdf_renderer_cmd) is None:
            raise RawTextOcrFailedError(
                file_path.stem,
                detail=f"pdf_renderer_not_found:{self.pdf_renderer_cmd}",
            )
        output_prefix = output_dir / f"page-{page_index + 1}"
        try:
            result = subprocess.run(
                [
                    self.pdf_renderer_cmd,
                    "-f",
                    str(page_index + 1),
                    "-l",
                    str(page_index + 1),
                    "-png",
                    "-singlefile",
                    "-r",
                    str(self.pdf_render_dpi),
                    str(file_path),
                    str(output_prefix),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.ocr_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RawTextOcrFailedError(
                file_path.stem,
                detail=f"pdf_render_failed:{error}",
            ) from error
        image_path = output_prefix.with_suffix(".png")
        if result.returncode != 0 or not image_path.exists():
            detail = result.stderr.strip() or f"exit_code={result.returncode}"
            raise RawTextOcrFailedError(
                file_path.stem,
                detail=f"pdf_render_failed:{detail}",
            )
        return [image_path]

    def inspect(self, doc_name: str) -> PdfInspectionMetrics:
        """Inspect PDF text/image resources without running OCR."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = PdfReader(str(file_path))
        _, metrics = self._extract_text_and_metrics(reader)
        return metrics

    def _extract_text_and_metrics(self, reader: PdfReader) -> tuple[str, PdfInspectionMetrics]:
        """Extract native text while collecting PDF resource metrics."""
        texts: list[str] = []
        native_text_chars = 0
        pages_with_images = 0
        pages_with_fonts = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            native_text_chars += len(text.strip())
            if text:
                texts.append(text)

            resources = page.get("/Resources") or {}
            if resources.get("/Font"):
                pages_with_fonts += 1
            if self._page_has_image_xobject(resources):
                pages_with_images += 1

        metrics = PdfInspectionMetrics(
            page_count=len(reader.pages),
            native_text_chars=native_text_chars,
            pages_with_images=pages_with_images,
            pages_with_fonts=pages_with_fonts,
        )
        return "\n".join(texts), metrics

    def _resolve_file_path(self, doc_name: str) -> Path:
        normalized_name = (
            doc_name
            if doc_name.lower().endswith(".pdf")
            else f"{doc_name}.pdf"
        )
        return self.base_dir / normalized_name

    def _env_flag(self, name: str, *, default: bool = False) -> bool:
        value = os.environ.get(name, "").strip().casefold()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    def _page_has_image_xobject(self, resources: object) -> bool:
        if not hasattr(resources, "get"):
            return False
        xobjects = resources.get("/XObject") or {}
        if not xobjects:
            return False
        for obj in xobjects.values():
            try:
                resolved = obj.get_object()
            except Exception:
                continue
            if resolved.get("/Subtype") == "/Image":
                return True
        return False

    def _build_ocr_provenance(
        self,
        *,
        doc_name: str,
        file_path: Path,
        page_count: int,
        schema_version: int = 1,
    ) -> dict[str, object]:
        return {
            "schema_version": schema_version,
            "doc_name": doc_name,
            "source_file_name": file_path.name,
            "source_file_sha256": self._file_sha256(file_path),
            "ocr_engine": "tesseract",
            "ocr_engine_version": self._tesseract_version(doc_name),
            "ocr_language": self.ocr_language,
            "ocr_page_limit": self.ocr_page_limit,
            "page_count": page_count,
            "pdf_renderer": self.pdf_renderer_cmd,
            "pdf_render_dpi": self.pdf_render_dpi,
        }

    def _file_sha256(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _reader_page_labels(self, reader: PdfReader) -> list[str | None]:
        try:
            labels = reader.page_labels or []
        except Exception:
            labels = []
        return [
            None if label is None else str(label)
            for label in labels
        ]

    def _tesseract_version(self, doc_name: str) -> str:
        try:
            result = subprocess.run(
                [self.tesseract_cmd, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.ocr_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_version_failed:{error}",
            ) from error

        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit_code={result.returncode}"
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_version_failed:{detail}",
            )
        return result.stdout.splitlines()[0].strip() if result.stdout.strip() else "unknown"
