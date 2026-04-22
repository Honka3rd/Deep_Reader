from collections.abc import Callable

from bundle_factory import BundleFactory
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from retrieval.faiss_index_bundle import FaissIndexBundle
from fingerprint_handler import FingerprintHandler
from config.faiss_storage_config import FaissStorageConfig


class BundleProvider:
    """Assemble runtime dependencies and return ready query bundles."""
    storage_config_factory: Callable[[str], FaissStorageConfig]
    fingerprint_handler_factory: Callable[[], FingerprintHandler]
    bundle_factory_provider: Callable[..., BundleFactory]
    loader_factory: DocumentLoaderFactory

    def __init__(
        self,
        storage_config_factory: Callable[[str], FaissStorageConfig],
        fingerprint_handler_factory: Callable[[], FingerprintHandler],
        bundle_factory_provider: Callable[..., BundleFactory],
        loader_factory: DocumentLoaderFactory,
    ):
        """Initialize provider from injected runtime factories/providers."""
        self.storage_config_factory = storage_config_factory
        self.fingerprint_handler_factory = fingerprint_handler_factory
        self.bundle_factory_provider = bundle_factory_provider
        self.loader_factory = loader_factory

    def _build_runtime_objects(
        self,
        doc_name: str,
    ) -> tuple[FaissStorageConfig, FingerprintHandler, BundleFactory]:
        """Build per-request runtime objects for a specific document."""
        config: FaissStorageConfig = self.storage_config_factory(doc_name)
        fingerprint_handler: FingerprintHandler
        fingerprint_handler = self.fingerprint_handler_factory()

        bundle_factory: BundleFactory = self.bundle_factory_provider(
            fingerprint_handler=fingerprint_handler,
        )

        return config, fingerprint_handler, bundle_factory

    def get_bundle(self, doc_name: str) -> FaissIndexBundle:
        """Ensure index/profile readiness and return query-ready bundle."""
        loader = self.loader_factory.get(doc_name)
        raw_text: str = loader.load(doc_name)
        return self.get_bundle_from_raw_text(
            doc_name=doc_name,
            raw_text=raw_text,
        )

    def get_bundle_from_raw_text(
        self,
        doc_name: str,
        raw_text: str,
    ) -> FaissIndexBundle:
        """Ensure index/profile readiness and return query-ready bundle from canonical raw text."""
        config, _, bundle_factory = self._build_runtime_objects(doc_name)
        return bundle_factory.ensure_index_ready(
            config=config,
            raw_text=raw_text,
        )
