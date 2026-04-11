from fingerprint_handler import FingerprintHandler
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from bundle_factory import BundleFactory
from container import ApplicationLookupContainer
from app_DI_config import AppDIConfig
from storage_config import StorageConfig
from faiss_index_bundle import FaissIndexBundle

class Coordinator:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str

    def __init__(
            self,
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            embedding_model: str = "text-embedding-3-small",
            llm_model: str = "gpt-4.1-mini",
            embedding_batch_size: int = 64,
            bundle_cache_capacity: int = 3,
    ):
        app_config = AppDIConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            llm_model=llm_model,
            embedding_batch_size=embedding_batch_size,
            bundle_cache_capacity=bundle_cache_capacity,
        )

        self.container = ApplicationLookupContainer.build(app_config)
        self.loader_factory = DocumentLoaderFactory()

    def _build_runtime_objects(
        self,
        doc_name: str,
        configuration: AppDIConfig | None = None,
    ) -> tuple[StorageConfig, FingerprintHandler, BundleFactory]:
        config: StorageConfig = self.container.storage_config_factory(doc_name)
        fingerprint_handler: FingerprintHandler
        if configuration is None:
             fingerprint_handler = (
                self.container.fingerprint_handler_factory()
            )
        else:
            fingerprint_handler = (
                self.container.fingerprint_handler_factory(
                    embedding_model=config.embedding_model,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            )

        bundle_factory: BundleFactory = self.container.bundle_factory_provider(
            fingerprint_handler=fingerprint_handler,
        )

        return config, fingerprint_handler, bundle_factory

    def get_bundle(self, doc_name: str, configuration: AppDIConfig | None = None) -> FaissIndexBundle:
        loader = self.loader_factory.get(doc_name)
        raw_text: str = loader.load(doc_name)

        config, _, bundle_factory = self._build_runtime_objects(doc_name, configuration)

        return bundle_factory.ensure_index_ready(
            config=config,
            raw_text=raw_text,
        )

