from collections.abc import Callable

from bundle_factory import BundleFactory
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from faiss_index_bundle import FaissIndexBundle
from fingerprint_handler import FingerprintHandler
from storage_config import StorageConfig


class BundleProvider:
    """Assemble runtime dependencies and return ready query bundles."""
    storage_config_factory: Callable[[str], StorageConfig]
    fingerprint_handler_factory: Callable[[], FingerprintHandler]
    bundle_factory_provider: Callable[..., BundleFactory]
    loader_factory: DocumentLoaderFactory

    def __init__(
        self,
        storage_config_factory: Callable[[str], StorageConfig],
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
    ) -> tuple[StorageConfig, FingerprintHandler, BundleFactory]:
        """Build per-request runtime objects for a specific document."""
        config: StorageConfig = self.storage_config_factory(doc_name)
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

        config, _, bundle_factory = self._build_runtime_objects(doc_name)
        return bundle_factory.ensure_index_ready(
            config=config,
            raw_text=raw_text,
        )
