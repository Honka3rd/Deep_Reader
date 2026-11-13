from pathlib import Path
from dataclasses import dataclass, field, replace
from uuid import uuid4
import hashlib
import json
import logging
from contextlib import nullcontext
from typing import Callable

from logging_config import log_stage

from bundle_provider import BundleProvider
from config.faiss_storage_config import FaissStorageConfig
from config.structured_document_storage_config import StructuredDocumentStorageConfig
from doc_loaders.document_load_errors import (
    RawTextOcrFailedError,
    RawTextOcrLowQualityError,
    RawTextRequiresOcrError,
)
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_extraction_contract import (
    PdfExtractionMode,
)
from doc_loaders.pdf_page_evidence import (
    PdfPageBoundaryEvidenceResult,
    PdfPageLayoutEvidence,
    PdfPageTextBoundary,
)
from doc_loaders.pdf_outline import PdfOutlineResult
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.structured_document import StructuredDocument
from document_structure.section_splitter_selector import SectionSplitterMode
from document_structure.text_normalization import normalize_ocr_text
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from doc_loaders.extraction_identity import extraction_matches, read_extraction_record
from document_preparation.preparation_mode import PreparationMode
from document_preparation.prepared_document_result import PreparedDocumentResult
from fingerprint_handler import FingerprintHandler
from language.document_language_detector import DocumentLanguageDetector
from language.language_code import LanguageCodeResolver, LanguageCode
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from profile.post_structure_metadata_enricher import PostStructureMetadataEnricher
from profile.profile_publication import ProfilePublicationConflict
from config.profile_publication_factory import profile_publication_guard
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from retrieval.node_provider import NodeProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualStructureSourceEvidence:
    """Source evidence prepared for explicit manual-structure validation/reparse."""

    doc_name: str
    source_identity: str
    raw_text: str
    source_hash: str
    language: str | None = None
    page_boundaries: list[PdfPageTextBoundary] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Server-only capability, never accepted from or serialized into public DTOs.
    validate_current: Callable[[StructuredDocument], str] | None = field(default=None, repr=False, compare=False)


class DocumentPreparationPipeline:
    """First-round preparation pipeline contract for one document."""

    def inspect_enhancement_capacity(self, *, snapshot, toc_hint="auto", enhancement_intent="restore_source"):
        """Use the exact planner without extraction, model generation or storage writes."""
        from document_structure.enhancement_capacity import capacity_identity, capacity_projection
        from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
        from document_structure.common_parse_policy import TocHint
        from document_structure.structure_plan_contract import EnhancementIntent, TocObservation
        try:
            splitter, observed = self.structured_document_builder.enhancement_request_context(
                snapshot.canonical_text, TocHint(toc_hint))
            preview, prompt, budget = splitter.assemble_candidate_request(
                snapshot.canonical_text, LanguageCode(snapshot.language), EnhancementIntent(enhancement_intent),
                TocHint(toc_hint), TocObservation(observed))
        except EnhancementPromptBudgetError as error:
            return {"capacity": capacity_projection(error.decision, reason=str(error),
                                                     source_chars=snapshot.text_length), "fingerprint": None}
        fingerprint = capacity_identity(prompt, budget, splitter.resource_limits)
        return {"capacity": capacity_projection({**budget, "full_source": True,
                    "visible_source_chars": preview["source_length"]}, source_chars=snapshot.text_length),
                "fingerprint": fingerprint}

    def prepare_enhancement_cleanup(self, *, candidate, binding, baseline=None, provider=None, is_cancelled=None):
        from document_preparation.cleanup_handoff import prepare_enhancement_cleanup
        if provider is None:
            provider = self.cleanup_provider
        return prepare_enhancement_cleanup(candidate=candidate, binding=binding,
                                           baseline=baseline, provider=provider, is_cancelled=is_cancelled)

    def build_enhancement_candidate(self, *, snapshot, toc_hint="auto",
                                    enhancement_intent="restore_source"):
        """Internal in-memory build; deliberately bypass all prepare/load stages."""
        from document_preparation.structure_enhancement_candidate import build_structure_enhancement_candidate
        return build_structure_enhancement_candidate(
            snapshot=snapshot, builder=self.structured_document_builder,
            toc_hint=toc_hint, enhancement_intent=enhancement_intent,
        )

    def __init__(
        self,
        loader_factory: DocumentLoaderFactory,
        language_detector: DocumentLanguageDetector,
        structured_document_builder: StructuredDocumentBuilder,
        structured_document_store: StructuredDocumentStore,
        node_provider: NodeProvider,
        faiss_index_builder: FaissIndexBuilder,
        faiss_index_store: FaissIndexStore,
        fingerprint_handler: FingerprintHandler,
        profile_builder: DocumentProfileBuilder,
        profile_store: DocumentProfileStore,
        bundle_provider: BundleProvider,
        post_structure_enricher: PostStructureMetadataEnricher | None = None,
        cleanup_provider=None,
        identity_resolver=None,
        profile_guard_factory=profile_publication_guard,
    ):
        """Initialize pipeline with explicit dependencies for preparation steps."""
        self.identity_resolver = identity_resolver
        self.profile_guard_factory = profile_guard_factory
        self.loader_factory = loader_factory
        self.cleanup_provider = cleanup_provider
        self.language_detector = language_detector
        self.structured_document_builder = structured_document_builder
        self.structured_document_store = structured_document_store
        self.node_provider = node_provider
        self.faiss_index_builder = faiss_index_builder
        self.faiss_index_store = faiss_index_store
        self.fingerprint_handler = fingerprint_handler
        self.profile_builder = profile_builder
        self.profile_store = profile_store
        self.bundle_provider = bundle_provider
        self.post_structure_enricher = (
            post_structure_enricher or PostStructureMetadataEnricher()
        )

    def _structured_config(self, namespace):
        return StructuredDocumentStorageConfig(namespace, identity_resolver=self.identity_resolver)

    def _index_config(self, namespace):
        return FaissStorageConfig(namespace, identity_resolver=self.identity_resolver)

    @log_stage(logger, "prepare_pipeline")
    def prepare(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.FREE_QA,
        structured_parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
        *,
        pdf_extraction_mode: PdfExtractionMode | str | None = None,
        toc_hint: str | None = None,
        enhancement_intent: str = "restore_source",
    ) -> PreparedDocumentAssets:
        """Bound loader evidence/budgets to one preparation, including lazy layout."""
        from document_structure.common_parse_policy import TocHint
        from document_structure.structure_plan_contract import EnhancementIntent, validate_enhancement_intent
        intent = validate_enhancement_intent(SectionSplitterMode.resolve(structured_parser_mode).value, enhancement_intent)
        hint_options = {"toc_hint": TocHint(toc_hint)} if toc_hint is not None else {}
        if intent != EnhancementIntent.RESTORE_SOURCE:
            hint_options["enhancement_intent"] = intent.value
        if self.identity_resolver is not None:
            doc_name = self.identity_resolver(doc_name)
        if (PreparationMode.resolve(mode) == PreparationMode.FREE_QA
                and not force_rebuild and pdf_extraction_mode is None
                and SectionSplitterMode.resolve(structured_parser_mode) == SectionSplitterMode.COMMON
                and toc_hint is None
                and self.structured_document_store.exists(self._structured_config(doc_name))):
            assets, _ = self._accepted_assets(doc_name)
            if assets.structured_document_ready:
                assets.faiss_ready, assets.faiss_namespace = self._prepare_faiss(
                    doc_name, assets.raw_text, assets, False)
                if assets.faiss_ready:
                    assets.bundle_ready = self._prepare_bundle(doc_name, assets.raw_text, assets, False)
            self._log_preparation_result(assets, reused=True)
            return assets
        scope = getattr(self.loader_factory, "preparation_scope", nullcontext)
        with scope():
            loader = self.loader_factory.get(doc_name)
            open_source = getattr(loader, "open_source", None)
            with log_stage(logger, "source_capture"):
                session = open_source(doc_name, mode=pdf_extraction_mode) if callable(open_source) else None
            assets = self._prepare_in_scope(doc_name, force_rebuild, mode, structured_parser_mode,
                source_session=session, **hint_options)
        return assets

    def _prepare_in_scope(
        self,
        doc_name: str,
        force_rebuild: bool,
        mode: PreparationMode | str,
        structured_parser_mode: SectionSplitterMode | str,
        *, source_session=None, toc_hint=None, enhancement_intent="restore_source",
    ) -> PreparedDocumentAssets:
        """Prepare a document in fixed order and return asset readiness snapshot.

        Args:
            doc_name: Logical document name.
            force_rebuild: When ``True``, skip artifact reuse and rebuild.
            mode: Preparation mode. ``base`` prepares only canonical/structured assets;
                ``free_qa`` prepares full QA runtime assets.
            structured_parser_mode: Structured parser mode used by structured build step.
        """
        preparation_mode = PreparationMode.resolve(mode)
        parser_mode = SectionSplitterMode.resolve(structured_parser_mode)
        logger.info("prepare_started doc_name=%s mode=%s parser_mode=%s force_rebuild=%s storage=%s",
                    json.dumps(doc_name, ensure_ascii=False), preparation_mode.value,
                    parser_mode.value, force_rebuild, type(self.structured_document_store).__name__)
        assets = PreparedDocumentAssets(
            doc_name=doc_name,
            raw_text=None,
            language=None,
            structured_document_ready=False,
            faiss_ready=False,
            profile_ready=False,
            bundle_ready=False,
            structured_document_path=None,
            faiss_namespace=None,
            errors=[],
            requested_toc_hint=toc_hint,
            enhancement_intent=enhancement_intent,
        )
        assets.source_session = source_session
        if source_session is not None:
            assets.apply_extraction_status(source_session.compatibility_status("pending"))

        if (
            preparation_mode == PreparationMode.BASE
            and self._try_reuse_existing_structured_document(
                doc_name=doc_name,
                assets=assets,
                force_rebuild=force_rebuild,
                parser_mode=parser_mode,
            )
        ):
            self._log_preparation_result(assets, reused=True)
            return assets

        assets.profile_creation_allowed = force_rebuild or not self.structured_document_store.exists(
            self._structured_config(namespace=doc_name))

        # Step 1. Load canonical raw text.
        extraction = None
        if source_session is None:
            raw_text, page_boundaries = self._load_raw_source(doc_name=doc_name, assets=assets)
        else:
            with log_stage(logger, "source_extract", extractor=source_session.identity.extractor_id):
                extraction = source_session.extract()
            raw_text = extraction.result.canonical_text
            page_boundaries = list(extraction.page_boundaries)
            if page_boundaries and self._validate_manual_structure_page_boundaries(boundaries=page_boundaries, raw_text=raw_text):
                raise ValueError("Invalid extracted text/page evidence")
            assets.extraction_result = extraction.result
            if not self._verify_source(assets):
                return assets
            assets.apply_extraction_status(source_session.compatibility_status("applied"))
        assets.raw_text = raw_text

        # Step 2. Detect document language.
        assets.language = self._detect_language(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            assets=assets,
        )

        # Step 3. Prepare profile artifact.
        prepared_profile = None
        if parser_mode != SectionSplitterMode.LLM_ENHANCED:
            assets.profile_ready, prepared_profile = self._prepare_profile(
                doc_name=doc_name, raw_text=assets.raw_text, language=assets.language,
                assets=assets, force_rebuild=force_rebuild)

        if assets.profile_publication_rejected:
            return assets

        # Step 4. Prepare structured document artifact.
        (
            assets.structured_document_ready,
            assets.structured_document_path,
        ) = self._prepare_structured_document(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            language=assets.language,
            document_profile=prepared_profile,
            assets=assets,
            force_rebuild=force_rebuild,
            parser_mode=parser_mode,
            page_boundaries=page_boundaries,
            extraction=extraction,
        )

        if parser_mode == SectionSplitterMode.LLM_ENHANCED:
            if not assets.structured_document_ready:
                self._log_preparation_result(assets)
                return assets
            assets.profile_ready, prepared_profile = self._prepare_profile(
                doc_name=doc_name, raw_text=assets.raw_text, language=assets.language,
                assets=assets, force_rebuild=force_rebuild)

        if not assets.structured_document_ready or not self._verify_source(assets):
            return assets

        # Step 4.5 Enrich profile with hierarchy-grounded structured metadata.
        if (
            assets.profile_ready
            and assets.profile_generated
            and prepared_profile is not None
            and assets.structured_document_ready
            and assets.structured_document_path is not None
        ):
            enriched_profile = self._enrich_profile_with_structured_document(
                doc_name=doc_name,
                profile=prepared_profile,
                structured_document_path=assets.structured_document_path,
                assets=assets,
            )
            if enriched_profile is not None:
                prepared_profile = enriched_profile

        if assets.profile_publication_rejected:
            return assets

        if preparation_mode == PreparationMode.BASE:
            self._log_preparation_result(assets)
            return assets

        # Step 5. Prepare FAISS artifacts.
        assets.faiss_ready, assets.faiss_namespace = self._prepare_faiss(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            assets=assets,
            force_rebuild=force_rebuild,
        )

        # Step 6. Prepare runtime bundle.
        if assets.faiss_ready:
            assets.bundle_ready = self._prepare_bundle(
                doc_name=doc_name, raw_text=assets.raw_text, assets=assets,
                force_rebuild=False,
            )

        self._log_preparation_result(assets)
        return assets

    @staticmethod
    def _log_preparation_result(assets: PreparedDocumentAssets, reused: bool = False) -> None:
        logger.info(
            "prepare_result doc_name=%s structured_reused=%s raw_chars=%s language=%s "
            "structured_ready=%s profile_ready=%s faiss_ready=%s bundle_ready=%s error_codes=%s",
            json.dumps(assets.doc_name, ensure_ascii=False), reused, len(assets.raw_text or ""),
            assets.language, assets.structured_document_ready, assets.profile_ready,
            assets.faiss_ready, assets.bundle_ready,
            json.dumps([error.split(":", 1)[0] for error in assets.errors]),
        )

    def prepare_and_load(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.FREE_QA,
        structured_parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
        *, toc_hint: str | None = None, enhancement_intent: str = "restore_source",
        pdf_extraction_mode: PdfExtractionMode | str | None = None,
    ) -> PreparedDocumentResult:
        """Prepare document artifacts, then load reusable runtime artifacts."""
        preparation_mode = PreparationMode.resolve(mode)
        if (preparation_mode == PreparationMode.FREE_QA and not force_rebuild
                and pdf_extraction_mode is None and toc_hint is None
                and SectionSplitterMode.resolve(structured_parser_mode) == SectionSplitterMode.COMMON):
            if self.identity_resolver is not None:
                doc_name = self.identity_resolver(doc_name)
            assets, document = self._accepted_assets(doc_name)
            if assets.structured_document_ready:
                assets.bundle_ready = self._prepare_bundle(doc_name, assets.raw_text, assets, False)
                assets.faiss_ready = assets.bundle_ready
                if assets.faiss_ready:
                    assets.faiss_namespace = self._index_config(doc_name).get_doc_name()
            self._log_preparation_result(assets, reused=True)
            return PreparedDocumentResult(assets=assets, structured_document=document,
                                          bundle=assets.runtime_bundle)
        assets = self.prepare(
            doc_name=doc_name,
            force_rebuild=force_rebuild,
            mode=preparation_mode,
            structured_parser_mode=structured_parser_mode,
            **({"pdf_extraction_mode": pdf_extraction_mode} if pdf_extraction_mode is not None else {}),
            **({"toc_hint": toc_hint} if toc_hint is not None else {}),
            **({"enhancement_intent": enhancement_intent} if enhancement_intent != "restore_source" else {}),
        )

        structured_document = None
        if assets.structured_document_ready and assets.structured_document_path is not None:
            try:
                if self.structured_document_store.exists(
                    assets.structured_document_path
                ):
                    with log_stage(logger, "structured_runtime_read"):
                        structured_document = self.structured_document_store.load(
                            assets.structured_document_path
                        )
                else:
                    assets.errors.append(
                        "prepare_and_load_structured_missing:"
                        f"{assets.structured_document_path}"
                    )
            except Exception as error:
                assets.errors.append(
                    f"prepare_and_load_structured_failed:{doc_name}:{error}"
                )

        bundle = assets.runtime_bundle if assets.bundle_ready else None
        if bundle is not None and not self._verify_source(assets):
            bundle = None

        return PreparedDocumentResult(
            assets=assets,
            structured_document=structured_document,
            bundle=bundle,
        )

    @staticmethod
    def _verify_source(assets):
        try:
            guard = assets.source_guard or assets.source_session
            if guard is not None:
                guard.verify_source()
            if assets.profile_publication_guard is not None:
                assets.profile_publication_guard.verify_source()
            return True
        except Exception as error:
            assets.errors.append(f"source_not_ready:explicit_prepare_required:{error}")
            assets.structured_document_ready = False
            assets.faiss_ready = False
            assets.bundle_ready = False
            assets.runtime_bundle = None
            assets.profile_ready = False
            return False

    def _accepted_assets(self, doc_name):
        from document_preparation.accepted_source_guard import AcceptedSourceGuard
        assets = PreparedDocumentAssets(doc_name, None, None, False, False, False, False, None, None)
        document = None
        try:
            config = self._structured_config(doc_name)
            if not self.structured_document_store.exists(config):
                raise ValueError("accepted_document_missing")
            document = self.structured_document_store.load(config)
            loader = self.loader_factory.get(doc_name)
            guard = AcceptedSourceGuard(
                document=document, doc_name=doc_name,
                source_path=lambda: loader.source_path(doc_name),
                current_document=lambda: self.structured_document_store.load(config))
            assets.source_guard = guard
            assets.raw_text = guard.snapshot.canonical_text
            assets.language = guard.snapshot.language
            assets.structured_document_ready = True
            assets.structured_document_path = self.structured_document_store.location(config)
            profile_config = self._index_config(doc_name)
            if self.profile_store.exists(profile_config):
                self.profile_store.load(profile_config)
                assets.profile_ready = True
        except Exception as error:
            assets.errors.append(f"runtime_not_ready:explicit_prepare_required:{error}")
        return assets, document

    def load_task_layout_page_boundaries(
        self, doc_name: str, *, document: StructuredDocument,
    ) -> list[PdfPageTextBoundary]:
        """Read proven persisted spans for this exact projection snapshot; never extract."""
        with log_stage(logger, "task_layout.persisted_page_evidence") as timing:
            timing["outcome"] = "unavailable"
            try:
                from document_preparation.accepted_source_guard import resolve_snapshot
                # No source resolution is needed when accepted identity is absent.
                read_extraction_record(document.parse_provenance)
                loader = self.loader_factory.get(doc_name)
                if not callable(getattr(loader, "source_path", None)):
                    timing["reason"] = "source_verification_unavailable"
                    return []
                snapshot = resolve_snapshot(document, doc_name, loader.source_path(doc_name))
                identity = snapshot.extraction_identity
                raw_text = snapshot.canonical_text
                spans = document.parse_provenance.get("source_page_spans", document.parse_provenance.get("pdf_page_spans"))
                if not isinstance(spans, list) or not spans:
                    timing["reason"] = "missing_spans"
                    return []
                boundaries = []
                for span in spans:
                    if not isinstance(span, dict) or any(
                        type(span.get(key)) is not int
                        for key in ("page_index", "char_start", "char_end")
                    ):
                        raise ValueError("invalid span fields")
                    label = span.get("page_label")
                    if label is not None and not isinstance(label, str):
                        raise ValueError("invalid page label")
                    boundaries.append(PdfPageTextBoundary(
                        page_index=span["page_index"], char_start=span["char_start"],
                        char_end=span["char_end"], page_label=label,
                        text=raw_text[span["char_start"]:span["char_end"]],
                        source_sha256=identity.source_sha256,
                    ))
                if self._validate_manual_structure_page_boundaries(
                    boundaries=boundaries, raw_text=raw_text,
                ):
                    timing["reason"] = "invalid_spans"
                    return []
                timing.update(outcome="reused", page_count=len(boundaries))
                return boundaries
            except (OSError, ValueError, TypeError, KeyError) as error:
                timing.update(reason="invalid_or_unavailable_evidence", error_type=type(error).__name__)
                return []

    def load_manual_structure_source_evidence(
        self,
        doc_name: str,
    ) -> ManualStructureSourceEvidence:
        """Share bounded source evidence within one explicit manual read."""
        scope = getattr(self.loader_factory, "preparation_scope", nullcontext)
        with scope():
            return self._load_manual_source_in_scope(doc_name)

    def _load_manual_source_in_scope(self, doc_name: str) -> ManualStructureSourceEvidence:
        """Load source evidence for explicit manual-structure reparse.

        This is a read-only handoff for app-layer manual reparse orchestration. It
        does not build profiles, write structured artifacts, prepare FAISS, or
        mutate task-layout state.
        """
        normalized_doc_name = doc_name.strip()
        if not normalized_doc_name:
            raise ValueError("doc_name cannot be empty")

        loader = self.loader_factory.get(normalized_doc_name)
        if callable(getattr(loader, "source_path", None)):
            config = self._structured_config(namespace=normalized_doc_name)
            if self.structured_document_store.exists(config):
                document = self.structured_document_store.load(config)
                if "extraction" in document.parse_provenance or "pdf_extraction" in document.parse_provenance:
                    from document_preparation.accepted_source_guard import AcceptedSourceGuard
                    from document_preparation.accepted_source_snapshot import SourceNotReady
                    guard = AcceptedSourceGuard(document=document, doc_name=normalized_doc_name,
                        source_path=lambda: loader.source_path(normalized_doc_name),
                        current_document=lambda: self.structured_document_store.load(config))
                    snapshot = guard.snapshot
                    reading_revision = document.parse_provenance.get("reading_content", {}).get("revision", 0)

                    def validate_current(current):
                        try:
                            accepted = guard.verify_document(current)
                        except SourceNotReady:
                            raise
                        except (OSError, ValueError) as error:
                            raise SourceNotReady("source_unavailable") from error
                        if accepted.cleanup_context != snapshot.cleanup_context:
                            raise SourceNotReady("accepted_page_or_ocr_evidence_changed")
                        if current.parse_provenance.get("reading_content", {}).get("revision", 0) != reading_revision:
                            raise SourceNotReady("reading_revision_changed")
                        return accepted.canonical_text
                    identity = snapshot.extraction_identity
                    raw_text = snapshot.canonical_text
                    spans = document.parse_provenance.get("source_page_spans", document.parse_provenance.get("pdf_page_spans", []))
                    if not isinstance(spans, list):
                        raise ValueError("Prepared page evidence is unavailable; prepare explicitly.")
                    try:
                        if any(not isinstance(span, dict) or any(type(span.get(key)) is not int
                               for key in ("page_index", "char_start", "char_end")) for span in spans):
                            raise ValueError("invalid span fields")
                        boundaries = [PdfPageTextBoundary(
                            **span, text=raw_text[span["char_start"]:span["char_end"]],
                            source_sha256=identity.source_sha256,
                        ) for span in spans]
                    except (KeyError, TypeError) as error:
                        raise ValueError("Prepared page evidence is invalid") from error
                    error = self._validate_manual_structure_page_boundaries(boundaries=boundaries, raw_text=raw_text) if boundaries else None
                    if error:
                        raise ValueError(f"Prepared PDF page evidence is invalid: {error}")
                    return ManualStructureSourceEvidence(
                        doc_name=normalized_doc_name, source_identity=normalized_doc_name,
                        raw_text=raw_text,
                        source_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                        language=document.language, page_boundaries=boundaries,
                        validate_current=validate_current,
                    )
                raise ValueError("Accepted extraction identity is unavailable; prepare explicitly.")
        raw_text = loader.load(normalized_doc_name)
        if raw_text is None or not raw_text.strip():
            raise ValueError(
                f"manual_structure source is empty for doc_name='{normalized_doc_name}'"
            )

        errors: list[str] = []
        language: str | None = None
        try:
            detected_language = self.language_detector.detect(
                normalize_ocr_text(raw_text)
            )
            language = detected_language.strip().lower() if detected_language else None
            if not language:
                errors.append("manual_structure_detect_language_empty_result")
        except Exception as error:
            errors.append(f"manual_structure_detect_language_failed:{error}")

        page_boundaries = self._load_manual_structure_page_boundaries(
            loader=loader,
            doc_name=normalized_doc_name,
            raw_text=raw_text,
            errors=errors,
        )

        return ManualStructureSourceEvidence(
            doc_name=normalized_doc_name,
            source_identity=normalized_doc_name,
            raw_text=raw_text,
            source_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            language=language,
            page_boundaries=page_boundaries,
            errors=errors,
        )

    def _load_manual_structure_page_boundaries(
        self,
        *,
        loader: object,
        doc_name: str,
        raw_text: str,
        errors: list[str],
    ) -> list[PdfPageTextBoundary]:
        """Load and validate optional page evidence for manual anchors."""
        try:
            load_page_boundary_evidence = getattr(
                loader,
                "load_page_boundary_evidence",
                None,
            )
            if callable(load_page_boundary_evidence):
                evidence = load_page_boundary_evidence(doc_name)
                boundaries = (
                    list(evidence.boundaries)
                    if isinstance(evidence, PdfPageBoundaryEvidenceResult)
                    else list(getattr(evidence, "boundaries", []) or [])
                )
            else:
                load_page_text_boundaries = getattr(
                    loader,
                    "load_page_text_boundaries",
                    None,
                )
                if not callable(load_page_text_boundaries):
                    return []
                boundaries = list(load_page_text_boundaries(doc_name) or [])
        except Exception as error:
            errors.append(f"manual_structure_page_boundaries_unavailable:{error}")
            return []

        validation_error = self._validate_manual_structure_page_boundaries(
            boundaries=boundaries,
            raw_text=raw_text,
        )
        if validation_error is not None:
            errors.append(f"manual_structure_page_boundaries_invalid:{validation_error}")
            return []
        return boundaries

    def _validate_manual_structure_page_boundaries(
        self,
        *,
        boundaries: list[PdfPageTextBoundary],
        raw_text: str,
    ) -> str | None:
        """Validate page evidence without promoting it to hierarchy authority."""
        if not boundaries:
            return None
        previous_end = 0
        seen_page_indices: set[int] = set()
        for expected_index, boundary in enumerate(boundaries):
            if boundary.page_index in seen_page_indices:
                return f"duplicate_page_index:{boundary.page_index}"
            seen_page_indices.add(boundary.page_index)
            if boundary.page_index != expected_index:
                return (
                    "non_contiguous_page_index:"
                    f"expected={expected_index},actual={boundary.page_index}"
                )
            if boundary.char_start < 0 or boundary.char_end < boundary.char_start:
                return f"invalid_range:{boundary.page_index}"
            if boundary.char_end > len(raw_text):
                return f"out_of_raw_text_range:{boundary.page_index}"
            if boundary.char_start < previous_end:
                return f"non_monotonic_range:{boundary.page_index}"
            if boundary.text and raw_text[boundary.char_start:boundary.char_end] != boundary.text:
                return f"text_mismatch:{boundary.page_index}"
            previous_end = boundary.char_end
        return None

    @log_stage(logger, "raw_load")
    def _load_raw_source(
        self,
        doc_name: str,
        assets: PreparedDocumentAssets,
    ) -> tuple[str | None, list[PdfPageTextBoundary] | None]:
        """Load text and optional same-pass page spans without shared mutable state."""
        try:
            loader = self.loader_factory.get(doc_name)
            logger.info("raw_loader_selected doc_name=%s loader=%s",
                        json.dumps(doc_name, ensure_ascii=False), type(loader).__name__)
            load_with_boundaries = getattr(loader, "load_with_page_boundary_evidence", None)
            if callable(load_with_boundaries):
                raw_text, evidence = load_with_boundaries(doc_name)
                page_boundaries = list(evidence.boundaries)
            else:
                raw_text = loader.load(doc_name)
                page_boundaries = None
            if not raw_text:
                assets.errors.append(
                    f"load_raw_text_empty:{doc_name}"
                )
                return None, None
            return raw_text, page_boundaries
        except RawTextRequiresOcrError:
            assets.errors.append(
                f"load_raw_text_requires_ocr:{doc_name}"
            )
            return None, None
        except RawTextOcrFailedError as error:
            assets.errors.append(
                f"load_raw_text_ocr_failed:{doc_name}:{error.detail or 'unknown'}"
            )
            return None, None
        except RawTextOcrLowQualityError as error:
            assets.errors.append(
                f"load_raw_text_ocr_low_quality:{doc_name}:{error.detail or 'unknown'}"
            )
            return None, None
        except Exception as error:
            assets.errors.append(
                f"load_raw_text_failed:{doc_name}:{error}"
            )
            return None, None

    @log_stage(logger, "language_detect")
    def _detect_language(
        self,
        doc_name: str,
        raw_text: str | None,
        assets: PreparedDocumentAssets,
    ) -> str | None:
        """Detect document language from raw text when text is available."""
        _ = doc_name
        if raw_text is None or not raw_text.strip():
            assets.errors.append("detect_language_skipped:missing_raw_text")
            return None

        try:
            language = self.language_detector.detect(normalize_ocr_text(raw_text))
            normalized_language = language.strip().lower() if language else None
            if not normalized_language:
                assets.errors.append("detect_language_empty_result")
                return None
            return normalized_language
        except Exception as error:
            assets.errors.append(f"detect_language_failed:{error}")
            return None

    @log_stage(logger, "structured_prepare")
    def _prepare_structured_document(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        document_profile: DocumentProfile | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
        parser_mode: SectionSplitterMode,
        page_boundaries: list[PdfPageTextBoundary] | None = None,
        extraction=None,
    ) -> tuple[bool, str | None]:
        """Build and persist structured document artifact."""
        if raw_text is None or not raw_text.strip():
            assets.errors.append("prepare_structured_document_skipped:missing_raw_text")
            return False, None
        if language is None:
            assets.errors.append("prepare_structured_document_skipped:missing_language")
            return False, None

        language_code = LanguageCodeResolver.resolve(language)
        if language_code == LanguageCode.UNKNOWN:
            assets.errors.append(
                f"prepare_structured_document_unsupported_language:{language}"
            )
            return False, None

        storage_config = self._structured_config(namespace=doc_name)
        structured_document_path = self.structured_document_store.location(storage_config)
        try:
            # TODO: Pass document_profile.parser_metadata hints into structured parser
            # once parser-side consumption is implemented.
            _ = document_profile
            if self._try_reuse_existing_structured_document(
                doc_name=doc_name,
                assets=assets,
                force_rebuild=force_rebuild,
                parser_mode=parser_mode,
                storage_config=storage_config,
                structured_document_path=structured_document_path,
            ):
                return True, structured_document_path

            if page_boundaries is None:
                page_boundaries = self._load_page_text_boundaries(doc_name)
            if extraction is None:
                outline = self._load_outline(doc_name)
            else:
                try:
                    outline = extraction.outline_provider() if extraction.outline_provider else None
                except Exception:
                    outline = None
            def layout_provider():
                if extraction is None:
                    return self._load_page_layout_evidence(doc_name)
                try:
                    return extraction.layout_provider() if extraction.layout_provider else None
                except Exception as error:
                    logger.warning("source_optional_layout_failed error_type=%s", type(error).__name__)
                    return None
            with log_stage(logger, "structured_build", raw_chars=len(raw_text), parser_mode=parser_mode.value) as timing:
                structured_document = self.structured_document_builder.build(
                    document_id=doc_name,
                    title=doc_name,
                    raw_text=raw_text,
                    language=language_code,
                    parser_mode=parser_mode,
                    page_evidence_provider=layout_provider,
                    page_boundaries=page_boundaries,
                    outline=outline,
                    **({"toc_hint": assets.requested_toc_hint} if assets.requested_toc_hint is not None else {}),
                    **({"enhancement_intent": assets.enhancement_intent} if assets.enhancement_intent != "restore_source" else {}),
                )
                timing["chapter_count"] = len(structured_document.chapters)
                timing["section_count"] = sum(len(chapter.sections) for chapter in structured_document.chapters)
                timing["structure_error_code"] = structured_document.structure_error_code
            if (
                parser_mode == SectionSplitterMode.LLM_ENHANCED
                and structured_document.structure_error_code is not None
            ):
                assets.errors.append(
                    "prepare_structured_document_llm_enhanced_rejected:"
                    f"{structured_document.structure_error_code}"
                )
                return False, None

            if assets.source_session is not None and assets.extraction_result is None:
                raise ValueError("Unproven extraction cannot be persisted")
            if assets.extraction_result is not None:
                result = assets.extraction_result
                if structured_document.raw_text != result.canonical_text:
                    raise ValueError("Extraction text changed before structured save")
                if not self._verify_source(assets):
                    return False, None
                if extraction is not None:
                    structured_document.parse_provenance.update(extraction.storage_metadata)
                    from document_preparation.cleanup_handoff import extraction_cleanup_metadata
                    structured_document.parse_provenance.update(extraction_cleanup_metadata(extraction))
                    if extraction.page_boundaries:
                        structured_document.parse_provenance["source_page_spans"] = [
                            {"page_index": b.page_index, "char_start": b.char_start,
                             "char_end": b.char_end, "page_label": b.page_label}
                            for b in extraction.page_boundaries]
                structured_document.parse_provenance["extraction"] = result.record()
            saved_document = self._save_structured_document_atomically(
                document=structured_document,
                storage_config=storage_config,
            )
            if assets.profile_publication_guard is not None:
                assets.profile_publication_guard.accept_prepared_structure(saved_document)
            self._persist_ocr_run(
                doc_name=doc_name,
                raw_text=raw_text,
                assets=assets,
                extraction=extraction,
            )
            return True, structured_document_path
        except Exception as error:
            assets.errors.append(f"prepare_structured_document_failed:{error}")
            return False, None

    @log_stage(logger, "structured_cache_check")
    def _try_reuse_existing_structured_document(
        self,
        *,
        doc_name: str,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
        parser_mode: SectionSplitterMode,
        storage_config: StructuredDocumentStorageConfig | None = None,
        structured_document_path: str | None = None,
    ) -> bool:
        """Reuse a valid structured document before any expensive raw load work."""
        if force_rebuild or parser_mode == SectionSplitterMode.LLM_ENHANCED:
            logger.info("structured_cache outcome=bypass force_rebuild=%s parser_mode=%s", force_rebuild, parser_mode.value)
            return False

        try:
            resolved_storage_config = storage_config or self._structured_config(
                namespace=doc_name
            )
            resolved_path = structured_document_path or self.structured_document_store.location(
                resolved_storage_config
            )
            if not self.structured_document_store.exists(resolved_storage_config):
                logger.info("structured_cache outcome=miss")
                return False
            with log_stage(logger, "structured_cache_read"):
                document = self.structured_document_store.load(resolved_storage_config)
            if assets.requested_toc_hint is not None:
                from document_structure.common_parse_policy import COMMON_POLICY_VERSION
                policy = document.parse_provenance.get("common_parse_policy", {})
                if (policy.get("version") != COMMON_POLICY_VERSION
                        or policy.get("toc_hint") != assets.requested_toc_hint
                        or policy.get("parser_mode") != parser_mode.value
                        or policy.get("enhancement_intent") != assets.enhancement_intent):
                    logger.info("structured_cache outcome=miss reason=common_policy_mismatch")
                    return False
            session = assets.source_session
            if session is not None:
                canonical_text = document.raw_text
                if not canonical_text:
                    from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
                    snapshot = resolve_accepted_source_snapshot(
                        document=document, namespace=doc_name,
                        expected_document_id=document.document_id,
                        structure_version=document.parse_provenance.get("storage_structure_version"),
                        source_path=session.source_path,
                        accepted=True,
                    )
                    canonical_text = snapshot.canonical_text
                if not extraction_matches(document.parse_provenance, session.identity, canonical_text):
                    logger.info("structured_cache outcome=miss reason=extraction_identity_mismatch")
                    return False
                session.verify_source()
                assets.apply_extraction_status(session.compatibility_status("reused"))
        except Exception as reuse_error:
            logger.warning("structured_cache outcome=reload_failed error_type=%s", type(reuse_error).__name__)
            assets.errors.append(
                f"prepare_structured_document_reload_failed:{doc_name}:{reuse_error}"
            )
            return False

        assets.structured_document_ready = True
        assets.structured_document_path = resolved_path
        logger.info("structured_cache outcome=hit")
        return True

    @log_stage(logger, "ocr_persist")
    def _persist_ocr_run(
        self,
        *,
        doc_name: str,
        raw_text: str,
        assets: PreparedDocumentAssets,
        extraction=None,
    ) -> None:
        """Persist OCR provenance after the document row exists; never controls parsing."""
        loader = self.loader_factory.get(doc_name)
        diagnostic = extraction.ocr_run if extraction is not None else None
        provenance = diagnostic.get("provenance") if diagnostic else (getattr(loader, "last_ocr_provenance", None) if extraction is None else None)
        save_ocr_run = getattr(self.structured_document_store, "save_ocr_run", None)
        if not isinstance(provenance, dict) or not provenance or not callable(save_ocr_run):
            return
        try:
            save_ocr_run(
                doc_name=doc_name,
                provenance=provenance,
                full_text=raw_text,
                pages=diagnostic.get("pages") if diagnostic else getattr(loader, "last_ocr_pages", None),
            )
        except Exception as error:
            assets.errors.append(f"persist_ocr_run_failed:{doc_name}:{error}")

    @log_stage(logger, "pdf_layout_evidence")
    def _load_page_layout_evidence(
        self,
        doc_name: str,
    ) -> list[PdfPageLayoutEvidence] | None:
        """Load bounded PDF layout evidence without making it parser authority."""
        try:
            loader = self.loader_factory.get(doc_name)
            load_evidence = getattr(loader, "load_page_layout_evidence", None)
            if load_evidence is None:
                return None
            return load_evidence(doc_name, candidate_page_limit=30)
        except Exception as error:
            # Layout analysis is enrichment; raw text preparation remains usable.
            logger.warning("pdf_layout_evidence outcome=unavailable error_type=%s", type(error).__name__)
            return None

    @log_stage(logger, "pdf_page_boundaries")
    def _load_page_text_boundaries(
        self,
        doc_name: str,
    ) -> list[PdfPageTextBoundary] | None:
        """Load page spans for exact TOC page-to-character projection."""
        try:
            loader = self.loader_factory.get(doc_name)
            load_boundaries = getattr(loader, "load_page_text_boundaries", None)
            if load_boundaries is None:
                return None
            return load_boundaries(doc_name)
        except Exception:
            return None

    @log_stage(logger, "pdf_outline")
    def _load_outline(self, doc_name: str) -> PdfOutlineResult | None:
        """Load native PDF outline evidence before visual TOC analysis."""
        try:
            loader = self.loader_factory.get(doc_name)
            load_outline = getattr(loader, "load_outline", None)
            if load_outline is None:
                return None
            return load_outline(doc_name)
        except Exception:
            return None

    @log_stage(logger, "profile_enrich")
    def _enrich_profile_with_structured_document(
        self,
        *,
        doc_name: str,
        profile: DocumentProfile,
        structured_document_path: str,
        assets: PreparedDocumentAssets,
    ) -> DocumentProfile | None:
        """Enrich profile with post-structure metadata without blocking prepare."""
        try:
            structured_path = Path(structured_document_path)
            if (
                not str(structured_document_path).startswith("postgres://")
                and not structured_path.exists()
            ):
                assets.errors.append(
                    f"post_structure_enrichment_missing_structured:{structured_document_path}"
                )
                return None

            structured_document = self.structured_document_store.load(
                structured_document_path
            )
            enriched_profile = self.post_structure_enricher.enrich(
                profile=profile,
                structured_document=structured_document,
            )
            guard = assets.profile_publication_guard
            if guard is None:
                raise ProfilePublicationConflict("profile_publication_binding_missing")
            with guard.publication() as validate:
                self.profile_store.save(enriched_profile, self._index_config(namespace=doc_name),
                                        validate=validate)
            print(
                "DocumentPreparationPipeline#post_structure_enrichment_success",
                f"doc={doc_name}",
                f"has_post_structure_metadata={enriched_profile.post_structure_metadata is not None}",
            )
            return enriched_profile
        except ProfilePublicationConflict as error:
            self._reject_profile_publication(assets, error)
            return None
        except Exception as error:
            print(
                "DocumentPreparationPipeline#post_structure_enrichment_failed",
                f"doc={doc_name}",
                f"error={error}",
            )
            assets.errors.append(
                f"post_structure_enrichment_failed:{doc_name}:{error}"
            )
            return None

    def _save_structured_document_atomically(
        self,
        *,
        document,
        storage_config: StructuredDocumentStorageConfig,
    ) -> StructuredDocument:
        """Atomically replace structured artifact to prevent half-written corruption."""
        if not document.parse_provenance.get("interaction_epoch"):
            document = replace(document, parse_provenance={
                **document.parse_provenance, "interaction_epoch": uuid4().hex})
        with log_stage(logger, "structured_save", storage=type(self.structured_document_store).__name__):
            self.structured_document_store.save(document=document, target=storage_config)
        # Read-back validation before reporting structured readiness.
        with log_stage(logger, "structured_readback"):
            return self.structured_document_store.load(storage_config)

    @log_stage(logger, "faiss_prepare")
    def _prepare_faiss(
        self,
        doc_name: str,
        raw_text: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> tuple[bool, str | None]:
        """Prepare FAISS index artifacts using canonical raw text."""
        if raw_text is None or not raw_text.strip():
            assets.errors.append("prepare_faiss_skipped:missing_raw_text")
            return False, None

        config = self._index_config(namespace=doc_name)
        namespace = config.get_doc_name()

        try:
            if not self._verify_source(assets):
                return False, None
            if not force_rebuild:
                try:
                    self.faiss_index_store.load_validated(config, raw_text, self.fingerprint_handler)
                    logger.info("faiss_cache outcome=hit")
                    return True, namespace
                except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                    pass

            logger.info("faiss_cache outcome=rebuild force_rebuild=%s", force_rebuild)
            with log_stage(logger, "retrieval_parse"):
                parsed_document = self.node_provider.parse(raw_text, config)
            with log_stage(logger, "faiss_build"):
                bundle = self.faiss_index_builder.build_from_parsed_document(parsed_document)
            with log_stage(logger, "faiss_save"):
                def verify():
                    if not self._verify_source(assets):
                        raise ValueError("source_not_ready:explicit_prepare_required")
                verify()
                self.faiss_index_store.publish(bundle, config, text=raw_text,
                    fingerprint=self.fingerprint_handler, verify_source=verify)
            return True, namespace
        except Exception as error:
            assets.errors.append(f"prepare_faiss_failed:{doc_name}:{error}")
            return False, None

    @log_stage(logger, "profile_prepare")
    def _prepare_profile(
        self, doc_name, raw_text, language, assets, force_rebuild,
    ) -> tuple[bool, DocumentProfile | None]:
        if force_rebuild:
            return self._prepare_profile_candidate(doc_name, raw_text, language, assets, True)
        # Initial cache creation retains cross-worker deduplication. This is only
        # a generation gate, not the source/lifecycle/SQL publication boundary.
        import fcntl
        config = self._index_config(namespace=doc_name)
        config.base_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(config.get_raw_profile_path()).with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._prepare_profile_candidate(doc_name, raw_text, language, assets, False)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _prepare_profile_candidate(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> tuple[bool, DocumentProfile | None]:
        """Prepare profile artifact using canonical raw text and detected language."""
        config = self._index_config(namespace=doc_name)

        try:
            if self.profile_store.exists(config) and not force_rebuild:
                logger.info("profile_cache outcome=hit")
                print(
                    "DocumentPreparationPipeline#profile_cache_hit",
                    f"doc={doc_name}",
                )
                try:
                    with log_stage(logger, "profile_load"):
                        profile = self.profile_store.load(config)
                    return True, profile
                except Exception as load_error:
                    assets.errors.append(
                        f"prepare_profile_reload_failed:{doc_name}:{load_error}"
                    )
                    return False, None

            if not (assets.profile_creation_allowed or force_rebuild):
                assets.errors.append("profile_not_ready:explicit_reparse_required")
                return False, None
            if not raw_text or not raw_text.strip():
                assets.errors.append("prepare_profile_skipped:missing_raw_text")
                return False, None
            if not language or not language.strip():
                assets.errors.append("prepare_profile_skipped:missing_language")
                return False, None

            print(
                "DocumentPreparationPipeline#profile_build_start",
                f"doc={doc_name}",
            )
            logger.info("profile_cache outcome=rebuild force_rebuild=%s", force_rebuild)
            guard = self.profile_guard_factory(name=doc_name, source=assets.source_session,
                store=self.structured_document_store, target=self._structured_config(doc_name),
                profile_config=config)
            guard.capture()
            assets.profile_publication_guard = guard
            with log_stage(logger, "profile_generate", raw_chars=len(raw_text), language=language.strip().lower()):
                profile = self.profile_builder.build(
                    text=raw_text,
                    document_language=language.strip().lower(),
                )
            with log_stage(logger, "profile_save"):
                with guard.publication() as validate:
                    self.profile_store.save(profile, config, validate=validate)
            assets.profile_generated = True
            print(
                "DocumentPreparationPipeline#profile_build_success",
                f"doc={doc_name}",
                f"has_parser_metadata={profile.parser_metadata is not None}",
                f"has_structure_profile={profile.structure_profile is not None}",
            )
            return True, profile
        except ProfilePublicationConflict as error:
            self._reject_profile_publication(assets, error)
            return False, None
        except Exception as error:
            print(
                "DocumentPreparationPipeline#profile_build_failed",
                f"doc={doc_name}",
                f"error={error}",
            )
            assets.errors.append(f"prepare_profile_failed:{doc_name}:{error}")
            return False, None

    @staticmethod
    def _reject_profile_publication(assets, error):
        assets.profile_publication_rejected = True
        assets.profile_ready = False
        assets.profile_generated = False
        assets.structured_document_ready = False
        assets.faiss_ready = False
        assets.bundle_ready = False
        assets.runtime_bundle = None
        assets.errors.append(f"source_not_ready:profile_publication_rejected:{error}")

    @log_stage(logger, "bundle_prepare")
    def _prepare_bundle(
        self,
        doc_name: str,
        raw_text: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> bool:
        """Validate runtime bundle readiness through the existing bundle provider path."""
        if raw_text is None or not raw_text.strip():
            assets.errors.append("prepare_bundle_skipped:missing_raw_text")
            return False

        try:
            if not self._verify_source(assets):
                return False
            bundle = self.bundle_provider.load_prepared_bundle(doc_name=doc_name, raw_text=raw_text)
            if not self._verify_source(assets):
                return False
            assets.runtime_bundle = bundle
            return bundle is not None
        except Exception as error:
            assets.errors.append(f"prepare_bundle_failed:{doc_name}:{error}")
            return False
