from pathlib import Path

from bundle_provider import BundleProvider
from config.faiss_storage_config import FaissStorageConfig
from config.structured_document_storage_config import StructuredDocumentStorageConfig
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.section_splitter_selector import SectionSplitterMode
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.preparation_mode import PreparationMode
from document_preparation.prepared_document_result import PreparedDocumentResult
from fingerprint_handler import FingerprintHandler
from language.document_language_detector import DocumentLanguageDetector
from language.language_code import LanguageCodeResolver, LanguageCode
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from retrieval.node_provider import NodeProvider


class DocumentPreparationPipeline:
    """First-round preparation pipeline contract for one document."""

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
    ):
        """Initialize pipeline with explicit dependencies for preparation steps."""
        self.loader_factory = loader_factory
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

    def prepare(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.FREE_QA,
        structured_parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
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
        )

        # Step 1. Load canonical raw text.
        raw_text = self._load_raw_text(
            doc_name=doc_name,
            assets=assets,
        )
        assets.raw_text = raw_text

        # Step 2. Detect document language.
        assets.language = self._detect_language(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            assets=assets,
        )

        # Step 3. Prepare structured document artifact.
        (
            assets.structured_document_ready,
            assets.structured_document_path,
        ) = self._prepare_structured_document(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            language=assets.language,
            assets=assets,
            force_rebuild=force_rebuild,
            parser_mode=parser_mode,
        )

        if preparation_mode == PreparationMode.BASE:
            return assets

        # Step 4. Prepare FAISS artifacts.
        assets.faiss_ready, assets.faiss_namespace = self._prepare_faiss(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            assets=assets,
            force_rebuild=force_rebuild,
        )

        # Step 5. Prepare profile artifact.
        assets.profile_ready = self._prepare_profile(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            language=assets.language,
            assets=assets,
            force_rebuild=force_rebuild,
        )

        # Step 6. Prepare runtime bundle.
        assets.bundle_ready = self._prepare_bundle(
            doc_name=doc_name,
            raw_text=assets.raw_text,
            assets=assets,
            force_rebuild=force_rebuild,
        )

        return assets

    def prepare_and_load(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.FREE_QA,
        structured_parser_mode: SectionSplitterMode | str = SectionSplitterMode.COMMON,
    ) -> PreparedDocumentResult:
        """Prepare document artifacts, then load reusable runtime artifacts."""
        preparation_mode = PreparationMode.resolve(mode)
        assets = self.prepare(
            doc_name=doc_name,
            force_rebuild=force_rebuild,
            mode=preparation_mode,
            structured_parser_mode=structured_parser_mode,
        )

        structured_document = None
        if assets.structured_document_ready and assets.structured_document_path is not None:
            try:
                structured_path = Path(assets.structured_document_path)
                if structured_path.exists():
                    structured_document = self.structured_document_store.load(
                        str(structured_path)
                    )
                else:
                    assets.errors.append(
                        f"prepare_and_load_structured_missing:{structured_path}"
                    )
            except Exception as error:
                assets.errors.append(
                    f"prepare_and_load_structured_failed:{doc_name}:{error}"
                )

        bundle = None
        if preparation_mode == PreparationMode.FREE_QA and assets.bundle_ready:
            try:
                if assets.raw_text is not None and assets.raw_text.strip():
                    bundle = self.bundle_provider.get_bundle_from_raw_text(
                        doc_name=doc_name,
                        raw_text=assets.raw_text,
                        force_rebuild=force_rebuild,
                    )
                else:
                    bundle = self.bundle_provider.get_bundle(doc_name)
            except Exception as error:
                assets.errors.append(
                    f"prepare_and_load_bundle_failed:{doc_name}:{error}"
                )

        return PreparedDocumentResult(
            assets=assets,
            structured_document=structured_document,
            bundle=bundle,
        )

    def _load_raw_text(
        self,
        doc_name: str,
        assets: PreparedDocumentAssets,
    ) -> str | None:
        """Load canonical raw text through document loader factory."""
        try:
            loader = self.loader_factory.get(doc_name)
            raw_text = loader.load(doc_name)
            if not raw_text:
                assets.errors.append(
                    f"load_raw_text_empty:{doc_name}"
                )
                return None
            return raw_text
        except Exception as error:
            assets.errors.append(
                f"load_raw_text_failed:{doc_name}:{error}"
            )
            return None

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
            language = self.language_detector.detect(raw_text)
            normalized_language = language.strip().lower() if language else None
            if not normalized_language:
                assets.errors.append("detect_language_empty_result")
                return None
            return normalized_language
        except Exception as error:
            assets.errors.append(f"detect_language_failed:{error}")
            return None

    def _prepare_structured_document(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
        parser_mode: SectionSplitterMode,
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

        storage_config = StructuredDocumentStorageConfig(namespace=doc_name)
        structured_document_path = storage_config.get_raw_document_path()
        try:
            should_rebuild = force_rebuild or parser_mode == SectionSplitterMode.LLM_ENHANCED
            if storage_config.exists() and not should_rebuild:
                try:
                    self.structured_document_store.load(storage_config)
                    return True, structured_document_path
                except Exception as load_error:
                    assets.errors.append(
                        f"prepare_structured_document_reload_failed:{doc_name}:{load_error}"
                    )

            structured_document = self.structured_document_builder.build(
                document_id=doc_name,
                title=doc_name,
                raw_text=raw_text,
                language=language_code,
                parser_mode=parser_mode,
            )
            self.structured_document_store.save(
                document=structured_document,
                target=storage_config,
            )
            return True, structured_document_path
        except Exception as error:
            assets.errors.append(f"prepare_structured_document_failed:{error}")
            return False, None

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

        config = FaissStorageConfig(namespace=doc_name)
        namespace = config.get_doc_name()

        try:
            has_position_metadata = self.faiss_index_store.has_position_metadata(config)
            fingerprint_matched = self.fingerprint_handler.matches(
                raw_text,
                config.get_raw_meta_path(),
            )
            if (
                not force_rebuild
                and config.exists()
                and has_position_metadata
                and fingerprint_matched
            ):
                return True, namespace

            self.faiss_index_store.clear(config)
            self.fingerprint_handler.clear(config.get_raw_meta_path())

            parsed_document = self.node_provider.parse(raw_text, config)
            bundle = self.faiss_index_builder.build_from_parsed_document(parsed_document)
            self.faiss_index_store.save(bundle, config)
            self.fingerprint_handler.save(raw_text, config.get_raw_meta_path())
            return True, namespace
        except Exception as error:
            assets.errors.append(f"prepare_faiss_failed:{doc_name}:{error}")
            return False, None

    def _prepare_profile(
        self,
        doc_name: str,
        raw_text: str | None,
        language: str | None,
        assets: PreparedDocumentAssets,
        force_rebuild: bool,
    ) -> bool:
        """Prepare profile artifact using canonical raw text and detected language."""
        if raw_text is None or not raw_text.strip():
            assets.errors.append("prepare_profile_skipped:missing_raw_text")
            return False
        if language is None or not language.strip():
            assets.errors.append("prepare_profile_skipped:missing_language")
            return False

        config = FaissStorageConfig(namespace=doc_name)

        try:
            if self.profile_store.exists(config) and not force_rebuild:
                try:
                    self.profile_store.load(config)
                    return True
                except Exception as load_error:
                    assets.errors.append(
                        f"prepare_profile_reload_failed:{doc_name}:{load_error}"
                    )
                    self.profile_store.clear(config)
            elif self.profile_store.exists(config) and force_rebuild:
                self.profile_store.clear(config)

            profile = self.profile_builder.build(
                text=raw_text,
                document_language=language.strip().lower(),
            )
            self.profile_store.save(profile, config)
            return True
        except Exception as error:
            assets.errors.append(f"prepare_profile_failed:{doc_name}:{error}")
            return False

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
            bundle = self.bundle_provider.get_bundle_from_raw_text(
                doc_name=doc_name,
                raw_text=raw_text,
                force_rebuild=force_rebuild,
            )
            return bundle is not None
        except Exception as error:
            assets.errors.append(f"prepare_bundle_failed:{doc_name}:{error}")
            return False
