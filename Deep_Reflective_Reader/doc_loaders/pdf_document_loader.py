from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace

from pypdf import PdfReader

from .abstract_document_loader import AbstractDocumentLoader
from .document_load_errors import RawTextOcrFailedError, RawTextRequiresOcrError
from .pdf_ocr_language_policy import normalize_tesseract_language_config
from .pdf_outline import PdfOutlineEntry, PdfOutlineResult
from .pdf_page_evidence import PdfPageLayoutEvidence, PdfPageTextBoundary, analyze_ocr_tsv


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


class PdfDocumentLoader(AbstractDocumentLoader):
    """Load PDF document files and join extracted page text."""
    base_dir: Path

    def __init__(
        self,
        base_dir: str = "data/raw",
        ocr_enabled: bool | None = None,
        ocr_language: str | None = None,
        tesseract_cmd: str | None = None,
        ocr_page_limit: int | None = None,
        ocr_timeout_seconds: int = 60,
        ocr_cache_enabled: bool | None = None,
        ocr_cache_dir: str | None = None,
        pdf_renderer_cmd: str | None = None,
        pdf_render_dpi: int = 200,
    ):
        """Initialize object state and injected dependencies.

Args:
    base_dir: Base dir.
"""
        self.base_dir = Path(base_dir)
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
        self.ocr_cache_enabled = (
            self._env_flag("DEEP_READER_PDF_OCR_CACHE_ENABLED", default=True)
            if ocr_cache_enabled is None
            else ocr_cache_enabled
        )
        self.ocr_cache_dir = (
            Path(ocr_cache_dir)
            if ocr_cache_dir is not None
            else self.base_dir.parent / "ocr_text"
        )
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
                    "ocr_started doc=%s language=%s page_limit=%s cache_enabled=%s",
                    doc_name,
                    self.ocr_language,
                    self.ocr_page_limit or len(reader.pages),
                    self.ocr_cache_enabled,
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
                        ocr_text=native_text,
                        analysis_stage="inventory",
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
                    )
                )
        return evidence

    def load_page_text_boundaries(self, doc_name: str) -> list[PdfPageTextBoundary]:
        """Return canonical page-to-character spans using the page-aware loader."""
        pages = self.load_pages(doc_name)
        separator = "\n\n"
        native_pages: list[str] = []
        try:
            reader = PdfReader(str(self._resolve_file_path(doc_name)))
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
                )
            )
            seen_native_page = seen_native_page or has_native_text
            if not has_native_text and page_index + 1 < len(pages):
                cursor += len(separator)
        return boundaries

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
    ) -> PdfPageLayoutEvidence:
        """Run coordinate OCR on one candidate page without changing raw text."""
        if shutil.which(self.tesseract_cmd) is None:
            return PdfPageLayoutEvidence(
                page_index=page_index,
                width=width,
                height=height,
                native_text_chars=native_text_chars,
                image_count=len(images),
                analysis_stage="inventory",
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
            )

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

        provenance = self._build_ocr_cache_provenance(
            doc_name=doc_name,
            file_path=file_path,
            page_count=len(reader.pages),
        )
        cached_text = self._load_cached_ocr_text(provenance)
        if cached_text is not None:
            self.last_ocr_provenance = provenance
            self.last_ocr_pages = None
            logger.info(
                "ocr_cache_hit doc=%s cache_path=%s text_chars=%s",
                doc_name,
                self._ocr_cache_path(provenance),
                len(cached_text),
            )
            return cached_text

        texts: list[str] = []
        page_limit = self.ocr_page_limit or len(reader.pages)
        with tempfile.TemporaryDirectory(prefix="deep-reader-pdf-ocr-") as temp_dir:
            temp_path = Path(temp_dir)
            for page_index, page in enumerate(reader.pages[:page_limit]):
                image_paths = self._render_page_images(
                    file_path=file_path,
                    page_index=page_index,
                    output_dir=temp_path,
                )
                for image_path in image_paths:
                    recognized_text = self._ocr_image_text(
                        image_path=image_path,
                        doc_name=doc_name,
                        page_index=page_index,
                    )
                    if recognized_text:
                        texts.append(recognized_text)
                if (page_index + 1) % 10 == 0 or page_index + 1 == page_limit:
                    logger.info(
                        "ocr_progress doc=%s page=%s/%s recognized_chars=%s",
                        doc_name,
                        page_index + 1,
                        page_limit,
                        sum(len(text) for text in texts),
                    )

        ocr_text = "\n\n".join(texts)
        if not ocr_text.strip():
            logger.error("ocr_failed doc=%s detail=empty_ocr_text", doc_name)
            raise RawTextOcrFailedError(doc_name=doc_name, detail="empty_ocr_text")
        self._write_cached_ocr_text(provenance=provenance, text=ocr_text)
        self.last_ocr_provenance = provenance
        self.last_ocr_pages = None
        logger.info(
            "ocr_completed doc=%s text_chars=%s cache_path=%s",
            doc_name,
            len(ocr_text),
            self._ocr_cache_path(provenance),
        )
        return ocr_text

    def _load_pages_with_ocr(
        self,
        *,
        doc_name: str,
        file_path: Path,
        reader: PdfReader,
    ) -> list[str]:
        """OCR one page at a time and cache page-preserving output."""
        if shutil.which(self.tesseract_cmd) is None:
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_not_found:{self.tesseract_cmd}",
            )

        provenance = self._build_ocr_cache_provenance(
            doc_name=doc_name,
            file_path=file_path,
            page_count=len(reader.pages),
            schema_version=2,
        )
        cached_pages = self._load_cached_ocr_pages(provenance)
        if cached_pages is not None:
            self.last_ocr_provenance = provenance
            self.last_ocr_pages = list(cached_pages)
            logger.info(
                "ocr_pages_cache_hit doc=%s cache_path=%s pages=%s",
                doc_name,
                self._ocr_cache_path(provenance),
                len(cached_pages),
            )
            return cached_pages

        pages: list[str] = []
        page_limit = self.ocr_page_limit or len(reader.pages)
        with tempfile.TemporaryDirectory(prefix="deep-reader-pdf-ocr-pages-") as temp_dir:
            temp_path = Path(temp_dir)
            for page_index, page in enumerate(reader.pages[:page_limit]):
                recognized_parts: list[str] = []
                image_paths = self._render_page_images(
                    file_path=file_path,
                    page_index=page_index,
                    output_dir=temp_path,
                )
                for image_path in image_paths:
                    recognized_text = self._ocr_image_text(
                        image_path=image_path,
                        doc_name=doc_name,
                        page_index=page_index,
                    )
                    if recognized_text:
                        recognized_parts.append(recognized_text)
                pages.append("\n\n".join(recognized_parts))

        if not any(page.strip() for page in pages):
            raise RawTextOcrFailedError(doc_name=doc_name, detail="empty_ocr_pages")
        self._write_cached_ocr_pages(provenance=provenance, pages=pages)
        self.last_ocr_provenance = provenance
        self.last_ocr_pages = list(pages)
        logger.info(
            "ocr_pages_completed doc=%s pages=%s text_chars=%s cache_path=%s",
            doc_name,
            len(pages),
            sum(len(page) for page in pages),
            self._ocr_cache_path(provenance),
        )
        return pages

    def _ocr_image_text(self, *, image_path: Path, doc_name: str, page_index: int) -> str:
        """Use the default segmentation first, then a vertical-text fallback."""
        outputs: list[str] = []
        for psm in (None, "5"):
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
            if text:
                outputs.append(text)
            if text:
                break
        return max(outputs, key=len, default="")

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

    def _build_ocr_cache_provenance(
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

    def _load_cached_ocr_pages(
        self,
        provenance: dict[str, object],
    ) -> list[str] | None:
        if not self.ocr_cache_enabled:
            return None
        cache_path = self._ocr_cache_path(provenance)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("provenance") != provenance:
            return None
        pages = payload.get("pages")
        if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
            return None
        return list(pages)

    def _write_cached_ocr_pages(
        self,
        *,
        provenance: dict[str, object],
        pages: list[str],
    ) -> None:
        if not self.ocr_cache_enabled:
            return
        cache_path = self._ocr_cache_path(provenance)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"provenance": provenance, "pages": pages}
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(cache_path.parent),
            delete=False,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_name = temp_file.name
        Path(temp_name).replace(cache_path)

    def _load_cached_ocr_text(self, provenance: dict[str, object]) -> str | None:
        if not self.ocr_cache_enabled:
            return None

        cache_path = self._ocr_cache_path(provenance)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("provenance") != provenance:
            return None

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return text

    def _write_cached_ocr_text(
        self,
        *,
        provenance: dict[str, object],
        text: str,
    ) -> None:
        if not self.ocr_cache_enabled:
            return

        cache_path = self._ocr_cache_path(provenance)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": provenance,
            "text": text,
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(cache_path.parent),
            delete=False,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_name = temp_file.name
        Path(temp_name).replace(cache_path)

    def _ocr_cache_path(self, provenance: dict[str, object]) -> Path:
        cache_key = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.ocr_cache_dir / f"{cache_key}.json"

    def _file_sha256(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
