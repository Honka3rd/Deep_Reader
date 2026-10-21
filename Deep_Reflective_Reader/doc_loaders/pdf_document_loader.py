from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from io import BytesIO
import logging
from logging_config import log_stage
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from collections.abc import Callable

from pypdf import PdfReader
from PIL import Image

from .abstract_document_loader import AbstractDocumentLoader
from .document_load_errors import (
    RawTextOcrFailedError,
    RawTextOcrLowQualityError,
    RawTextRequiresOcrError,
)
from .pdf_ocr_language_policy import normalize_tesseract_language_config
from .pdf_ocr_decision import PdfOcrDecision, PdfOcrPurpose
from .pdf_ocr_policy import PdfOcrPolicy
from .pdf_execution_pass import PdfExecutionLimits, PdfExecutionPass, PdfBudgetExceeded, in_pdf_pass
from .pdf_page_inspection import PdfPageInspection, inspect_pdf_page
from .pdf_outline import PdfOutlineEntry, PdfOutlineResult
from .pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason, PdfExtractionMode
from .pdf_extraction_result import PdfExtractionResult
from .pdf_fast_extraction import extract_fast
from .pdf_full_ocr_extraction import extract_full_ocr
from .pdf_page_evidence import (
    PdfPageBoundaryEvidenceResult,
    PdfPageLayoutEvidence,
    PdfPageTextBoundary,
    analyze_ocr_tsv,
)
from .raw_data_paths import resolve_raw_data_dir


logger = logging.getLogger(__name__)
OCR_QUALITY_VERSION = "observed-script-v3"


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
    region_evidence: tuple = ()


class PdfDocumentLoader(AbstractDocumentLoader):
    """Load PDF document files and join extracted page text."""
    supports_page_boundaries = True

    def source_path(self, doc_name: str):
        return self._resolve_file_path(doc_name)

    def render_source_region(self, doc_name, source_sha256, region):
        """Render a persisted review rectangle only; never run text extraction/OCR."""
        import io
        from PIL import Image
        path = self.source_path(doc_name)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != source_sha256:
            raise ValueError("review_source_changed")
        page = region["page_index"]
        dpi = region["render_dpi"]
        bbox = region["bbox"]
        if type(page) is not int or page < 0 or dpi != self.pdf_render_dpi or len(bbox)!=4:
            raise ValueError("review_geometry_unavailable")
        if not all(type(v) is int and v>=0 for v in bbox):
            raise ValueError("review_geometry_invalid")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > 20000000:
            raise ValueError("review_geometry_invalid")
        with tempfile.TemporaryDirectory(prefix="source-review-") as directory, self.preparation_scope():
            root = Path(directory)
            snapshot = root/"source.pdf"
            snapshot.write_bytes(source)
            images = self._render_page_images(file_path=snapshot,page_index=page,output_dir=root)
            if len(images)!=1:
                raise ValueError("review_render_failed")
            with Image.open(images[0]) as image:
                if bbox[2]>image.width or bbox[3]>image.height:
                    raise ValueError("review_geometry_invalid")
                output = io.BytesIO()
                image.crop(tuple(bbox)).save(output,format="PNG")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source_sha256:
                raise ValueError("review_source_changed")
            return output.getvalue()

    def open_source(self, doc_name: str, *, mode=None):
        from .source_session import SourceSession, SourceArtifact
        from .extraction_identity import normalize_pdf_identity
        from .pdf_extraction_identity import identity_for_loader
        path = self.source_path(doc_name)
        source = path.read_bytes()
        selected = PdfExtractionMode.FAST if mode is None else PdfExtractionMode(mode)
        identity = identity_for_loader(self, source_sha256=hashlib.sha256(source).hexdigest(), mode=selected)

        def extract():
            result = self.extract(doc_name, mode=selected, source_bytes=source)
            generic = result.as_extraction_result(identity)
            boundaries = tuple(result.boundary_evidence.boundaries)
            spans = [{"page_index": b.page_index, "char_start": b.char_start,
                      "char_end": b.char_end, "page_label": b.page_label} for b in boundaries]
            diagnostic = None
            if result.provenance:
                diagnostic = {"provenance": {**result.provenance,
                    "source_file_sha256": identity.source_sha256,
                    "source_file_name": result.boundary_evidence.source_file_name,
                    "ocr_engine": "tesseract", "pdf_renderer": self.pdf_renderer_cmd,
                    "pdf_render_dpi": self.pdf_render_dpi,
                    "page_quality": list(result.page_quality)}, "pages": list(result.pages)}
            return SourceArtifact(generic, boundaries, result.outline_provider, result.layout_provider,
                {"pdf_extraction": identity.record(result.text), "pdf_page_spans": spans,
                 **({"source_review":result.provenance["source_review"]} if "source_review" in result.provenance else {})}, diagnostic)

        return SourceSession(identity=normalize_pdf_identity(identity), source_path=path, extract=extract,
            status=lambda state: {"requested_pdf_extraction_mode": selected,
                "effective_pdf_extraction_mode": selected if state != "pending" else None,
                "pdf_extraction_identity": identity,
                "pdf_extraction_provenance": state if state != "pending" else "unknown"})
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
        execution_limits: PdfExecutionLimits | None = None,
        ocr_policy: PdfOcrPolicy | None = None,
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
        if ocr_page_limit is not None and (type(ocr_page_limit) is not int or ocr_page_limit < 0):
            raise ValueError("ocr_page_limit must be a non-negative integer or None")
        self.ocr_timeout_seconds = ocr_timeout_seconds
        self.pdf_renderer_cmd = (
            pdf_renderer_cmd
            or os.environ.get("DEEP_READER_PDF_RENDERER_CMD", "pdftoppm").strip()
            or "pdftoppm"
        )
        self.pdf_render_dpi = int(
            os.environ.get("DEEP_READER_PDF_RENDER_DPI", str(pdf_render_dpi))
        )
        self.execution_limits = execution_limits or PdfExecutionLimits()
        self.ocr_policy = ocr_policy or PdfOcrPolicy()
        self._pass_context = ContextVar("pdf_execution_pass", default=None)
        self._last_result_context = ContextVar("pdf_last_result", default=None)

    @contextmanager
    def preparation_scope(self):
        current = self._pass_context.get()
        if current is not None:
            yield current
            return
        state = PdfExecutionPass(self.execution_limits)
        token = self._pass_context.set(state)
        try:
            yield state
        finally:
            logger.info("pdf_pass_finished render_attempts=%s ocr_attempts=%s cache_entries=%s cache_bytes=%s",
                        sum(v for (_, stage), v in state.counters.items() if stage == "render"),
                        sum(v for (_, stage), v in state.counters.items() if stage == "ocr"),
                        len(state.cache), state.cache_bytes)
            state.clear_cache()
            self._pass_context.reset(token)
            self._last_result_context.set(state)

    def _result_state(self):
        state = self._pass_context.get() or self._last_result_context.get()
        if state is None:
            state = PdfExecutionPass(self.execution_limits)
            self._last_result_context.set(state)
        return state

    @property
    def last_ocr_provenance(self):
        return self._result_state().provenance

    @last_ocr_provenance.setter
    def last_ocr_provenance(self, value):
        self._result_state().provenance = value

    @property
    def last_ocr_pages(self):
        return self._result_state().pages

    @last_ocr_pages.setter
    def last_ocr_pages(self, value):
        self._result_state().pages = value

    @property
    def last_ocr_quality(self):
        return self._result_state().quality

    @last_ocr_quality.setter
    def last_ocr_quality(self, value):
        self._result_state().quality = value

    def _inspect_page(self, page, *, source_sha256, page_index):
        state = self._pass_context.get()
        key = ("inspection-v2", source_sha256, page_index)
        inspection = state.get(key, stage="native", page=page_index)
        if inspection is None:
            with log_stage(logger, "pdf_native_inspection", page=page_index):
                inspection = inspect_pdf_page(page, source_sha256=source_sha256, page_index=page_index)
            state.put(key, inspection)
        return inspection

    def _assess_page(self, page, purpose):
        state = self._pass_context.get()
        key = ("decision", page.source_sha256, page.page_index, purpose.value,
               self.ocr_policy.config.policy_version, page.evidence_schema_version)
        decision = state.get(key, stage="decision", page=page.page_index, purpose=purpose.value)
        if decision is None:
            decision = self.ocr_policy.assess(page, purpose)
            state.put(key, decision)
        logger.info("pdf_page_decision page=%s purpose=%s decision=%s reasons=%s policy=%s",
                    page.page_index, purpose.value, decision.decision.value,
                    ",".join(reason.value for reason in decision.reason_codes), decision.policy_version)
        return decision

    def _binary_identity(self, command):
        # Binary content fingerprints invalidate reuse across engine/renderer builds
        # without executing extra version subprocesses on layout-only paths.
        try:
            resolved = shutil.which(command)
            if not resolved:
                return None
            path = Path(resolved).resolve()
            stat = path.stat()
            key = ("binary", str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            state = self._pass_context.get()
            identity = state.get(key, stage="binary_identity")
            if identity is None:
                identity = (str(path), self._file_sha256(path))
                state.put(key, identity)
            return identity
        except OSError:
            return None

    def _execution_key(self, source_hash, page, purpose, rotation=0):
        engine = self._binary_identity(self.tesseract_cmd)
        renderer = self._binary_identity(self.pdf_renderer_cmd)
        if not source_hash or engine is None or renderer is None or rotation is None:
            return None
        return ("pdf-execution-v1", OCR_QUALITY_VERSION, source_hash, page, purpose,
                self.ocr_policy.config.policy_version, 1, engine, renderer,
                self.ocr_language, self.pdf_render_dpi, self.ocr_enabled,
                self.ocr_timeout_seconds, "full-page", rotation)

    def _run_process(self, command, *, purpose, stage, page):
        return self._pass_context.get().run(
            subprocess.run, command, purpose=purpose, stage=stage, page=page,
            timeout=self.ocr_timeout_seconds,
        )

    def extract(
        self, doc_name: str, *, mode: PdfExtractionMode | str = PdfExtractionMode.FAST,
        source_bytes: bytes | None = None,
    ) -> PdfExtractionResult:
        """Explicit mode entry; legacy load methods keep their existing contract."""
        selected = PdfExtractionMode(mode)
        executors = {
            PdfExtractionMode.FAST: extract_fast,
            PdfExtractionMode.OCR: partial(extract_full_ocr, loader=self, quality_version=OCR_QUALITY_VERSION),
            PdfExtractionMode.AUTO: self._extract_auto,
        }
        executor = executors.get(selected)
        if executor is None:
            raise PdfExtractionError(
                PdfExtractionFailureReason.MODE_UNAVAILABLE,
                f"PDF {selected.value} extraction is not integrated yet.",
            )
        file_path = self._resolve_file_path(doc_name)
        source = file_path.read_bytes() if source_bytes is None else source_bytes
        source_sha256 = hashlib.sha256(source).hexdigest()
        if selected in {PdfExtractionMode.OCR, PdfExtractionMode.AUTO}:
            executor = partial(executor, source=source)
        try:
            reader = PdfReader(BytesIO(source))
            return executor(
                reader, doc_name=doc_name, source_file_name=file_path.name,
                source_sha256=source_sha256,
                outline_provider=lambda: self._read_native_outline(reader, source_sha256),
            )
        except PdfExtractionError:
            raise
        except Exception as error:
            raise PdfExtractionError(
                PdfExtractionFailureReason.EXTRACTION_FAILED,
                "Unable to extract PDF text; check for a corrupt or encrypted source.",
            ) from error

    def _extract_auto(self, reader, *, source, doc_name, source_file_name, source_sha256, outline_provider):
        from copy import copy

        def run(operation):
            with tempfile.TemporaryDirectory(prefix="deep-reader-auto-") as directory:
                snapshot = copy(self)
                snapshot.base_dir = Path(directory)
                (snapshot.base_dir / source_file_name).write_bytes(source)
                with snapshot.preparation_scope():
                    return operation(snapshot)

        try:
            def canonical(snapshot):
                text, evidence = snapshot.load_with_page_boundary_evidence(source_file_name)
                return text, evidence, snapshot.last_ocr_provenance, tuple(snapshot.last_ocr_quality or [])
            text, evidence, provenance, quality = run(canonical)
            if not text.strip():
                raise PdfExtractionError(PdfExtractionFailureReason.NO_TEXT, "Automatic extraction produced no text.")
        except RawTextOcrLowQualityError as error:
            raise PdfExtractionError(PdfExtractionFailureReason.QUALITY_FAILED, "Automatic OCR quality rejected.") from error
        except RawTextRequiresOcrError as error:
            raise PdfExtractionError(PdfExtractionFailureReason.ENGINE_UNAVAILABLE, "Automatic recovery requires enabled OCR.") from error
        except RawTextOcrFailedError as error:
            coverage = re.fullmatch(r"page=(\d+):native_coverage_unresolved", error.detail or "")
            if coverage:
                raise PdfExtractionError(
                    PdfExtractionFailureReason.EXTRACTION_FAILED,
                    f"Automatic recovery stopped at page {coverage[1]}: native_coverage_unresolved. "
                    "OCR could not preserve all native text. Retry fast for native text or "
                    "explicit OCR for a new transcription; no partial result was saved.",
                ) from error
            reason = PdfExtractionFailureReason.BUDGET_EXCEEDED if "budget" in (error.detail or "") or "limit" in (error.detail or "") else PdfExtractionFailureReason.EXTRACTION_FAILED
            raise PdfExtractionError(reason, "Automatic recovery failed.") from error
        pages = tuple(boundary.text for boundary in evidence.boundaries)
        return PdfExtractionResult(
            text=text, pages=pages, boundary_evidence=replace(evidence, doc_name=doc_name),
            effective_mode=PdfExtractionMode.AUTO, extraction_version="selective-pages-v1",
            empty_page_indexes=tuple(index for index, text in enumerate(pages) if not text),
            warnings=("automatic_graphics_coverage_limits_apply",),
            layout_provider=lambda: run(lambda snapshot: snapshot.load_page_layout_evidence(source_file_name, candidate_page_limit=30)),
            outline_provider=outline_provider, provenance=provenance or {}, page_quality=quality,
        )

    @in_pdf_pass
    def load(self, doc_name: str) -> str:
        """Load complete canonical text; never return a partially recovered PDF."""
        _, pages, _, separator = self._load_canonical_pages(doc_name)
        return separator.join(text for text in pages if text)

    @in_pdf_pass
    def load_pages(self, doc_name: str) -> list[str]:
        """Load canonical text while preserving one text value per PDF page."""
        _, pages, _, _ = self._load_canonical_pages(doc_name)
        return pages

    def _load_canonical_pages(self, doc_name: str) -> tuple[PdfReader, list[str], str, str]:
        try:
            file_path = self._resolve_file_path(doc_name)
            source = file_path.read_bytes()
            source_sha256 = hashlib.sha256(source).hexdigest()
            reader = PdfReader(BytesIO(source))
            policy = self.ocr_policy
            inspections = [
                self._inspect_page(page, source_sha256=source_sha256, page_index=index)
                for index, page in enumerate(reader.pages)
            ]
            pages = [page.native_text or "" for page in inspections]
            separator = "\n" if any(pages) else "\n\n"
            recovery_indexes = []
            for page in inspections:
                decision = self._assess_page(page, PdfOcrPurpose.TEXT_RECOVERY)
                if decision.decision is PdfOcrDecision.SKIP:
                    continue
                # Short/header-only native text is complete when the inspected
                # page has no other drawing content. Unknown inventory is not zero.
                if self._native_only_text_complete(page, policy):
                    logger.info("pdf_page_gate page=%s purpose=text_recovery reason=native_only_text_complete", page.page_index)
                    continue
                recovery_indexes.append(page.page_index)
            logger.info(
                "pdf_raw_pages_inspected doc=%s pages=%s native_text_chars=%s recovery_pages=%s",
                doc_name, len(pages), sum(len(text.strip()) for text in pages), len(recovery_indexes),
            )
            if not recovery_indexes:
                self._clear_ocr_result()
                return reader, pages, source_sha256, separator
            if not self.ocr_enabled:
                raise RawTextRequiresOcrError(
                    doc_name, detail=f"page={recovery_indexes[0] + 1}:text_recovery_unresolved",
                )
            def validate_recovery(index: int, text: str) -> None:
                native = inspections[index]
                # Full-page fallback replaces, never appends to, the native layer.
                # Reject OCR that drops usable native content instead of guessing
                # region order or publishing a header-only result.
                if native.native_text and native.text_quality is not None and policy._quality_ok(native.text_quality):
                    native_key = self._text_coverage_key(native.native_text)
                    recovered_key = self._text_coverage_key(text)
                    if native_key and (
                        native_key not in recovered_key
                        or (native.image_draw_count != 0 and native_key == recovered_key)
                    ):
                        raise RawTextOcrFailedError(
                            doc_name, detail=f"page={index + 1}:native_coverage_unresolved",
                        )

            recovered = self._load_pages_with_ocr(
                doc_name=doc_name, file_path=file_path, reader=reader,
                page_indexes=recovery_indexes, page_validator=validate_recovery,
            )
            for index in recovery_indexes:
                validate_recovery(index, recovered[index])
                pages[index] = recovered[index]
            if self._file_sha256(file_path) != source_sha256:
                raise RawTextOcrFailedError(doc_name, detail="source_changed_during_recovery")
            return reader, pages, source_sha256, separator
        except PdfBudgetExceeded as error:
            self._clear_ocr_result()
            self._pass_context.get().clear_cache()
            raise RawTextOcrFailedError(doc_name, detail=error.reason) from error
        except Exception:
            self._clear_ocr_result()
            self._pass_context.get().clear_cache()
            raise

    @staticmethod
    def _native_only_text_complete(page: PdfPageInspection, policy: PdfOcrPolicy) -> bool:
        return bool(
            page.native_text and page.native_text.strip() and page.text_quality is not None
            and policy._quality_ok(page.text_quality)
            and page.image_draw_count == 0 and page.image_coverage_ratio == 0
            and (page.has_vector_paint is False or page.has_only_page_background)
            and not (set(page.flags) - {"text_position_unavailable", "text_position_limit"})
            and page.has_font_resources is True
        )

    @staticmethod
    def _text_coverage_key(text: str) -> str:
        return "".join(char.casefold() for char in text if char.isalnum())

    def _clear_ocr_result(self) -> None:
        self.last_ocr_provenance = None
        self.last_ocr_pages = None
        self.last_ocr_quality = []

    @in_pdf_pass
    def load_page_layout_evidence(
        self,
        doc_name: str,
        *,
        candidate_page_limit: int | None = None,
    ) -> list[PdfPageLayoutEvidence]:
        """Collect native-first inventory, with optional OCR for visual candidates."""
        if candidate_page_limit is not None and (
            type(candidate_page_limit) is not int or candidate_page_limit < 0
        ):
            raise ValueError("candidate_page_limit must be a non-negative integer or None")
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        source = file_path.read_bytes()
        reader = PdfReader(BytesIO(source))
        source_sha256 = hashlib.sha256(source).hexdigest()
        evidence: list[PdfPageLayoutEvidence] = []
        page_limit = len(reader.pages) if candidate_page_limit is None else candidate_page_limit
        for page_index, page in enumerate(reader.pages):
            inspection = self._inspect_page(page, source_sha256=source_sha256, page_index=page_index)
            decision = self._assess_page(inspection, PdfOcrPurpose.LAYOUT_ANALYSIS)
            native_text = (inspection.native_text or "").strip()
            width, height = self._page_dimensions(page, [])
            inventory = PdfPageLayoutEvidence(
                page_index=page_index,
                width=width,
                height=height,
                native_text_chars=len(native_text),
                image_count=inspection.image_draw_count or 0,
                source_file_name=file_path.name,
                source_sha256=source_sha256,
                rotation_degrees=inspection.rotation_degrees,
                ocr_text=native_text,
                cache_key=self._layout_cache_key(
                    source_sha256=source_sha256, page_index=page_index, stage="inventory",
                ),
                cache_hit=False,
                evidence=[
                    "page_inventory", f"layout_ocr_decision:{decision.decision.value}",
                    f"layout_ocr_policy:{decision.policy_version}",
                    *[f"layout_ocr_reason:{reason.value}" for reason in decision.reason_codes],
                    *inspection.flags,
                    *(["image_count_unknown"] if inspection.image_draw_count is None else []),
                ],
            )
            # Uncertainty is not OCR permission. Preserve native-only fallback;
            # visual/rotated candidates retain the explicitly enabled legacy pass.
            visual_candidate = (
                bool(inspection.image_draw_count)
                or "form_or_unknown_xobject_uninspected" in inspection.flags
                or inspection.rotation_degrees in (90, 180, 270)
            )
            if not self.ocr_enabled:
                gate = "layout_ocr_disabled"
            elif page_index >= page_limit:
                gate = "layout_candidate_limit"
            elif decision.decision is PdfOcrDecision.SKIP:
                gate = "layout_ocr_not_needed"
            elif decision.decision is PdfOcrDecision.UNCERTAIN and not visual_candidate:
                gate = "layout_uncertain_inventory_fallback"
            else:
                gate = None
            if gate is not None:
                logger.info("pdf_page_gate page=%s purpose=layout_analysis reason=%s", page_index, gate)
                evidence.append(replace(inventory, evidence=[*inventory.evidence, gate]))
                continue
            if shutil.which(self.tesseract_cmd) is None:
                evidence.append(replace(inventory, failure_reason="tesseract_unavailable"))
                continue
            failure_reason = None
            state = self._pass_context.get()
            execution_key = self._execution_key(source_sha256, page_index, "layout_analysis", inspection.rotation_degrees)
            cache_key = (execution_key, file_path.name) if execution_key else None
            cached = state.get(cache_key, stage="coordinate_ocr", page=page_index, purpose="layout_analysis")
            if cached is not None:
                evidence.append(replace(cached, cache_hit=True))
                continue
            with tempfile.TemporaryDirectory(prefix="deep-reader-page-layout-render-") as temp_dir:
                try:
                    image_paths = self._render_page_images(
                        file_path=file_path,
                        page_index=page_index,
                        output_dir=Path(temp_dir),
                        purpose="layout_analysis",
                    )
                except PdfBudgetExceeded as error:
                    evidence.append(replace(inventory, failure_reason=error.reason))
                    continue
                except (RawTextOcrFailedError, OSError, subprocess.TimeoutExpired):
                    image_paths = []
                    failure_reason = "layout_renderer_failed"
                # Decode embedded images only if the selected page cannot render.
                images = []
                if not image_paths:
                    try:
                        images = list(page.images)
                    except Exception:
                        failure_reason = "layout_image_decode_failed"
                if not image_paths and not images:
                    evidence.append(replace(
                        inventory, failure_reason=failure_reason or "no_ocr_sources",
                    ))
                    continue
                try:
                    if image_paths:
                        with Image.open(image_paths[0]) as rendered:
                            layout_width, layout_height = rendered.size
                    else:
                        layout_width, layout_height = self._page_dimensions(page, images)
                    layout = self._build_page_layout_evidence(
                        page_index=page_index,
                        width=layout_width,
                        height=layout_height,
                        native_text_chars=len(native_text),
                        images=images,
                        image_paths=image_paths,
                        source_file_name=file_path.name,
                        source_sha256=source_sha256,
                    )
                except PdfBudgetExceeded as error:
                    evidence.append(replace(inventory, failure_reason=error.reason))
                    continue
                except Exception as exc:
                    # Optional evidence must not discard other pages or raw text.
                    evidence.append(replace(
                        inventory, failure_reason="layout_ocr_failed",
                        evidence=[*inventory.evidence, f"layout_failure_type:{type(exc).__name__}"],
                    ))
                    continue
                if layout.failure_reason or not layout.word_count:
                    evidence.append(replace(
                        inventory, failure_reason=layout.failure_reason or "layout_ocr_empty",
                    ))
                    continue
                evidence.append(replace(
                    layout,
                    cache_key=(
                        "pdf-layout:v2:" + hashlib.sha256(json.dumps(cache_key).encode()).hexdigest()
                        if cache_key is not None else None
                    ),
                    evidence=[
                        *inventory.evidence, *layout.evidence,
                        "layout_visual_candidate",
                        *([failure_reason] if failure_reason else []),
                    ],
                ))
                if self._file_sha256(file_path) == source_sha256:
                    if image_paths and failure_reason is None:
                        state.put(cache_key, evidence[-1])
                else:
                    evidence[-1] = replace(inventory, failure_reason="layout_source_changed")
        return evidence

    def load_page_text_boundaries(self, doc_name: str) -> list[PdfPageTextBoundary]:
        """Return canonical page-to-character spans using the page-aware loader."""
        evidence = self.load_page_boundary_evidence(doc_name)
        return list(evidence.boundaries)

    def load_page_boundary_evidence(self, doc_name: str) -> PdfPageBoundaryEvidenceResult:
        """Return compact page boundary evidence for manual TOC anchors."""
        _, evidence = self.load_with_page_boundary_evidence(doc_name)
        return evidence

    @in_pdf_pass
    def load_with_page_boundary_evidence(
        self, doc_name: str,
    ) -> tuple[str, PdfPageBoundaryEvidenceResult]:
        """Share one canonical extraction between preparation text and spans."""
        file_path = self._resolve_file_path(doc_name)
        reader, pages, source_sha256, separator = self._load_canonical_pages(doc_name)
        page_labels = self._reader_page_labels(reader)
        boundaries: list[PdfPageTextBoundary] = []
        cursor = 0
        seen_text = False
        for page_index, text in enumerate(pages):
            if text and seen_text:
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
            seen_text = seen_text or bool(text)
        evidence = PdfPageBoundaryEvidenceResult(
            doc_name=doc_name,
            source_file_name=file_path.name,
            source_sha256=source_sha256,
            page_count=len(reader.pages),
            boundaries=boundaries,
        )
        return separator.join(text for text in pages if text), evidence

    @in_pdf_pass
    def load_outline(self, doc_name: str) -> PdfOutlineResult:
        """Read and validate native PDF bookmarks without invoking OCR."""
        file_path = self._resolve_file_path(doc_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
        source = file_path.read_bytes()
        reader = PdfReader(BytesIO(source))
        source_sha256 = hashlib.sha256(source).hexdigest()
        return self._read_native_outline(reader, source_sha256)

    def _read_native_outline(self, reader: PdfReader, source_sha256: str) -> PdfOutlineResult:
        """Read bookmarks from the same snapshot as the selected extraction."""
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

    @in_pdf_pass
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
                    result = self._run_process(
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
                        purpose="layout_analysis", stage="ocr", page=page_index,
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

    @in_pdf_pass
    def _load_pages_with_ocr(
        self,
        *,
        doc_name: str,
        file_path: Path,
        reader: PdfReader,
        page_indexes: list[int] | None = None,
        page_validator: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        """OCR one page at a time and retain page-preserving output in memory."""
        indexes = list(range(len(reader.pages))) if page_indexes is None else list(page_indexes)
        page_limit = len(reader.pages) if self.ocr_page_limit is None else self.ocr_page_limit
        if any(index >= page_limit for index in indexes):
            raise RawTextOcrFailedError(doc_name, detail="ocr_page_limit_excludes_required_page")
        if shutil.which(self.tesseract_cmd) is None:
            raise RawTextOcrFailedError(
                doc_name=doc_name,
                detail=f"tesseract_not_found:{self.tesseract_cmd}",
            )

        state = self._pass_context.get()
        source_hash = self._file_sha256(file_path)
        execution_key = self._execution_key(
            source_hash, tuple(indexes), "text_recovery",
            tuple(reader.pages[index].get("/Rotate", 0) for index in indexes),
        )
        cache_key = (execution_key, doc_name, self.ocr_page_limit, "selective-pages-v1", 3) if execution_key else None
        cached = state.get(cache_key, stage="raw_pages", purpose="text_recovery")
        if cached is not None:
            pages, provenance, quality = cached
            if page_validator is not None:
                for index in indexes:
                    page_validator(index, pages[index])
            self.last_ocr_provenance, self.last_ocr_pages, self.last_ocr_quality = provenance, pages, quality
            return list(pages)

        provenance = self._build_ocr_provenance(
            doc_name=doc_name,
            file_path=file_path,
            page_count=len(reader.pages),
            schema_version=3,
        )
        provenance["recovery_page_indexes"] = indexes
        provenance["raw_recovery_version"] = "selective-pages-v1"
        provenance["policy_version"] = self.ocr_policy.config.policy_version

        self._clear_ocr_result()
        pages: list[str] = [""] * len(reader.pages)
        quality_by_page: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="deep-reader-pdf-ocr-pages-") as temp_dir:
            temp_path = Path(temp_dir)
            for page_index in indexes:
                recognized_parts: list[str] = []
                page_candidates: list[OcrCandidateQuality] = []
                with log_stage(logger, "pdf_page_render", doc_name=doc_name,
                               page=page_index + 1, page_total=page_limit, dpi=self.pdf_render_dpi):
                    image_paths = self._render_page_images(
                        file_path=file_path,
                        page_index=page_index,
                        output_dir=temp_path,
                    )
                if len(image_paths) != 1:
                    raise RawTextOcrFailedError(
                        doc_name, detail=f"page={page_index + 1}:expected_one_full_page_render",
                    )
                if self._render_is_uniform(image_paths[0]):
                    if page_validator is not None:
                        page_validator(page_index, "")
                    quality_by_page.append({
                        "page_index": page_index, "blank_render_confirmed": True,
                        "candidate_count": 0, "selected_passed": False,
                    })
                    continue
                for image_path in image_paths:
                    try:
                        recognized_text, candidates = self._ocr_image_text_with_quality(
                            image_path=image_path,
                            doc_name=doc_name,
                            page_index=page_index,
                        )
                    except (OSError, subprocess.TimeoutExpired) as error:
                        raise RawTextOcrFailedError(
                            doc_name, detail=f"page={page_index + 1}:ocr_process_failed:{type(error).__name__}",
                        ) from error
                    page_candidates.extend(candidates)
                    if recognized_text:
                        recognized_parts.append(recognized_text)
                pages[page_index] = "\n\n".join(recognized_parts)
                summary = self._summarize_page_ocr_quality(
                    page_index=page_index, candidates=page_candidates,
                )
                if not pages[page_index].strip() or not summary["selected_passed"]:
                    if any(candidate.text for candidate in page_candidates):
                        raise RawTextOcrLowQualityError(
                            doc_name, detail=f"page_quality_gate_failed:page={page_index + 1}",
                        )
                    raise RawTextOcrFailedError(doc_name, detail=f"empty_ocr_page:page={page_index + 1}")
                if page_validator is not None:
                    page_validator(page_index, pages[page_index])
                quality_by_page.append(summary)
                if (page_index + 1) % 10 == 0 or page_index + 1 == page_limit:
                    logger.info(
                        "ocr_progress doc=%s page=%s/%s recognized_chars=%s",
                        doc_name,
                        page_index + 1,
                        page_limit,
                        sum(len(page_text) for page_text in pages),
                    )

        if self._file_sha256(file_path) != provenance["source_file_sha256"]:
            raise RawTextOcrFailedError(doc_name, detail="source_changed_during_recovery")
        self.last_ocr_provenance = provenance
        self.last_ocr_pages = list(pages)
        self.last_ocr_quality = list(quality_by_page)
        state.put(cache_key, (pages, provenance, quality_by_page))
        logger.info(
            "ocr_pages_completed doc=%s pages=%s text_chars=%s persistence=memory_then_ocr_runs",
            doc_name,
            len(pages),
            sum(len(page) for page in pages),
        )
        return pages

    @staticmethod
    def _render_is_uniform(image_path: Path) -> bool:
        """A uniform full-page render is blank; failed OCR alone cannot prove it."""
        try:
            with Image.open(image_path) as rendered:
                if rendered.mode not in {"1", "L", "RGB"} or "transparency" in rendered.info:
                    return False
                extrema = rendered.convert("RGB").getextrema()
            return all(low == high for low, high in extrema)
        except (OSError, ValueError):
            return False

    def _ocr_image_text(self, *, image_path: Path, doc_name: str, page_index: int) -> str:
        """Return the best OCR candidate text for one rendered page image."""
        text, _ = self._ocr_image_text_with_quality(
            image_path=image_path,
            doc_name=doc_name,
            page_index=page_index,
        )
        return text

    @in_pdf_pass
    def _ocr_image_text_with_quality(
        self,
        *,
        image_path: Path,
        doc_name: str,
        page_index: int,
        allow_review: bool = False,
    ) -> tuple[str, list[OcrCandidateQuality]]:
        """Compare bounded Tesseract segmentation candidates deterministically."""
        from .chinese_ocr_strategy import recognize
        directed = recognize(self, image_path, doc_name=doc_name, page_index=page_index, allow_review=allow_review)
        if directed is not None:
            candidate = replace(self._score_ocr_candidate(psm="chinese_vertical_columns", text=directed.text),
                                region_evidence=directed.regions)
            return (directed.text if candidate.passed else ""), [candidate]
        candidates: list[OcrCandidateQuality] = []
        for psm in (None, "5", "6", "11"):
            command = [self.tesseract_cmd, str(image_path), "stdout", "-l", self.ocr_language]
            if psm is not None:
                command.extend(["--psm", psm])
            with log_stage(logger, "ocr_candidate", doc_name=doc_name, page=page_index + 1,
                           psm=psm or "default", language=self.ocr_language) as timing:
                result = self._run_process(command, purpose="text_recovery", stage="ocr", page=page_index)
                timing["return_code"] = result.returncode
                timing["recognized_chars"] = len(result.stdout.strip())
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
        selected = max(candidates, key=lambda candidate: (candidate.passed, candidate.score), default=None)
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
        # Installed language packs are capabilities, not evidence of page language.
        cjk_dominant = cjk_ratio >= 0.35

        # CJK OCR legitimately inserts spaces between characters/words. Test
        # fragmentation on the remaining lexical tokens, not CJK or punctuation.
        fragmentation_tokens = [
            token for token in tokens
            if any(char.isalnum() and not "\u3400" <= char <= "\u9fff" for char in token)
        ]
        fragmented_ratio = (
            sum(len(token) <= 2 for token in fragmentation_tokens) / len(fragmentation_tokens)
            if fragmentation_tokens else 0.0
        )

        if cjk_dominant:
            score = (
                (0.55 * cjk_ratio)
                + min(cjk_count / 120.0, 0.25)
                + min(non_space_count / 500.0, 0.1)
                - (0.35 * latin_ratio)
                - (0.25 * symbol_ratio)
                - (0.3 * fragmented_ratio)
            )
            threshold = 0.28
        else:
            score = (
                (0.45 * min((latin_count + digit_count) / 80.0, 1.0))
                + (0.25 * (1.0 - symbol_ratio))
                + min(non_space_count / 500.0, 0.15)
                - (0.25 * fragmented_ratio)
            )
            threshold = 0.22

        reasons: list[str] = []
        if not normalized:
            reasons.append("empty_text")
        if symbol_ratio > 0.35:
            reasons.append("symbol_heavy")
        if fragmented_ratio > 0.82 and len(fragmentation_tokens) >= 20:
            reasons.append("fragmented_tokens")
        if non_space_count < 20:
            reasons.append("too_short")
        latin_tokens = re.findall(r"[A-Za-z]+", normalized)
        fragmented_latin = (
            len(latin_tokens) >= 10
            and sum(len(token) <= 2 for token in latin_tokens) / len(latin_tokens) > 0.60
        )
        if fragmented_latin:
            reasons.append("fragmented_latin")
        if "\ufffd" in normalized or any(ord(char) < 32 and not char.isspace() for char in normalized):
            reasons.append("invalid_encoding")
        blocking_reasons = [reason for reason in reasons if reason != "too_short"]
        passed = signal_count >= 4 and score >= threshold and not blocking_reasons
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
        selected = max(candidates, key=lambda candidate: (candidate.passed, candidate.score), default=None)
        return {
            "quality_version": OCR_QUALITY_VERSION,
            "page_index": page_index,
            "selected_psm": selected.psm if selected is not None else None,
            "selected_score": selected.score if selected is not None else 0.0,
            "selected_passed": selected.passed if selected is not None else False,
            "candidate_count": len(candidates),
            "regions": list(selected.region_evidence) if selected is not None else [],
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

    @in_pdf_pass
    def _render_page_images(
        self,
        *,
        file_path: Path,
        page_index: int,
        output_dir: Path,
        purpose: str = "text_recovery",
    ) -> list[Path]:
        """Render one complete PDF page so OCR does not depend on image XObject decoding."""
        if shutil.which(self.pdf_renderer_cmd) is None:
            raise RawTextOcrFailedError(
                file_path.stem,
                detail=f"pdf_renderer_not_found:{self.pdf_renderer_cmd}",
            )
        output_prefix = output_dir / f"page-{page_index + 1}"
        try:
            result = self._run_process(
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
                purpose=purpose, stage="render", page=page_index,
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

    @in_pdf_pass
    def inspect_pages(self, doc_name: str) -> tuple[PdfPageInspection, ...]:
        """Collect native page evidence without rendering/OCR or changing load policy."""
        source = self._resolve_file_path(doc_name).read_bytes()
        source_sha256 = hashlib.sha256(source).hexdigest()
        reader = PdfReader(BytesIO(source))
        return tuple(
            self._inspect_page(page, source_sha256=source_sha256, page_index=index)
            for index, page in enumerate(reader.pages)
        )

    def _extract_text_and_metrics(self, reader: PdfReader) -> tuple[str, PdfInspectionMetrics]:
        """Extract native text while collecting PDF resource metrics."""
        pages, metrics = self._extract_pages_and_metrics(reader)
        return "\n".join(page_text for page_text in pages if page_text), metrics

    def _extract_pages_and_metrics(
        self,
        reader: PdfReader,
        *,
        strip_pages: bool = False,
    ) -> tuple[list[str], PdfInspectionMetrics]:
        """Extract native page text once while collecting PDF resource metrics."""
        texts: list[str] = []
        native_text_chars = 0
        pages_with_images = 0
        pages_with_fonts = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            if strip_pages:
                text = text.strip()
            native_text_chars += len(text.strip())
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
        return texts, metrics

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
            "quality_version": OCR_QUALITY_VERSION,
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

    @in_pdf_pass
    def _tesseract_version(self, doc_name: str) -> str:
        state = self._pass_context.get()
        identity = self._binary_identity(self.tesseract_cmd)
        key = ("engine_version", identity) if identity else None
        cached = state.get(key, stage="engine_version", purpose="text_recovery")
        if cached is not None:
            return cached
        try:
            result = self._run_process(
                [self.tesseract_cmd, "--version"],
                purpose="text_recovery", stage="metadata", page=-1,
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
        version = result.stdout.splitlines()[0].strip() if result.stdout.strip() else "unknown"
        if version != "unknown":
            state.put(key, version)
        return version
