from faiss_index_builder import FaissIndexBuilder
from faiss_index_bundle import FaissIndexBundle
from faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from storage_config import StorageConfig
from bundle_factory import BundleFactory

class QueryEngine:
    faiss_index_builder: FaissIndexBuilder
    faiss_index_store: FaissIndexStore
    bundle: FaissIndexBundle | None = None
    bundle_factory: BundleFactory
    def __init__(
        self,
        faiss_index_builder: FaissIndexBuilder,
        faiss_index_store: FaissIndexStore,
        fingerprint_handler: FingerprintHandler,
        bundle_factory: BundleFactory
    ):
        self.faiss_index_builder = faiss_index_builder
        self.faiss_index_store = faiss_index_store
        self.fingerprint_handler = fingerprint_handler
        self.bundle_factory = bundle_factory

    def ready(self, raw_text: str, config: StorageConfig) -> FaissIndexBundle:
        return self.bundle_factory.ensure_index_ready(config, raw_text)
