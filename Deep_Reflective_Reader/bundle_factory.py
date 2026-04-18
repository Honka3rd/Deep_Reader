from collections import OrderedDict
from typing import OrderedDict as OrderedDictType

from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from retrieval.node_provider import NodeProvider
from config.storage_config import StorageConfig
from profile.document_profile import DocumentProfile
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore


class BundleFactory:
    """Build, cache, reload, and invalidate per-document FAISS bundles."""
    builder: FaissIndexBuilder
    node_provider: NodeProvider
    fingerprint_handler: FingerprintHandler
    store: FaissIndexStore
    profile_builder: DocumentProfileBuilder
    profile_store: DocumentProfileStore
    cache_capacity: int
    bundle_cache: OrderedDictType[str, FaissIndexBundle]

    def __init__(
        self,
        builder: FaissIndexBuilder,
        fingerprint_handler: FingerprintHandler,
        store: FaissIndexStore,
        node_provider: NodeProvider,
        profile_builder: DocumentProfileBuilder,
        profile_store: DocumentProfileStore,
        cache_capacity: int = 3,
    ):
        """Initialize object state and injected dependencies.

Args:
    builder: Builder.
    fingerprint_handler: Fingerprint handler.
    store: Store.
    node_provider: Node provider.
    profile_builder: Profile builder.
    profile_store: Profile store.
    cache_capacity: Cache capacity.
"""
        self.builder = builder
        self.node_provider = node_provider
        self.fingerprint_handler = fingerprint_handler
        self.store = store
        self.profile_builder = profile_builder
        self.profile_store = profile_store
        self.cache_capacity = cache_capacity
        self.bundle_cache = OrderedDict()

    def invalidate(self, doc_name: str) -> None:
        """Remove a cached bundle entry for the specified document.

Args:
    doc_name: Logical document name and artifact namespace (without extension)."""
        self.bundle_cache.pop(doc_name, None)

    def clear(self) -> None:
        """Remove persisted artifacts for clean rebuild."""
        self.bundle_cache.clear()

    def _evict_if_needed(self) -> None:
        """Internal helper for evict if needed."""
        while len(self.bundle_cache) > self.cache_capacity:
            evicted_doc_name, _ = self.bundle_cache.popitem(last=False)
            print(f"BundleFactory#evict: evicted '{evicted_doc_name}' from cache")

    def _put_cache(self, doc_name: str, bundle: FaissIndexBundle) -> None:
        """Internal helper for put cache.

Args:
    doc_name: Logical document name and artifact namespace (without extension).
    bundle: Active FaissIndexBundle used for lookup and metadata access."""
        if doc_name in self.bundle_cache:
            self.bundle_cache.pop(doc_name)

        self.bundle_cache[doc_name] = bundle
        self._evict_if_needed()

    def clear_artifacts(self, config: StorageConfig, doc_name: str) -> None:
        """Remove persisted artifacts and cached bundle for a document.

Args:
    config: StorageConfig describing filesystem artifact paths.
    doc_name: Logical document name and artifact namespace (without extension)."""
        self.store.clear(config)
        self.profile_store.clear(config)
        self.fingerprint_handler.clear(config.get_raw_meta_path())
        self.invalidate(doc_name)

    def ensure_profile_ready(
        self,
        config: StorageConfig,
        raw_text: str,
        document_language: str,
    ) -> DocumentProfile:
        """Ensure document profile exists and rebuild when missing/corrupted.

Args:
    config: StorageConfig describing filesystem artifact paths.
    raw_text: Raw full document text before parsing/indexing.
    document_language: Primary document language code (e.g. en/zh).

Returns:
    Loaded or newly generated document profile for the target document."""
        if self.profile_store.exists(config):
            try:
                print(
                    f"BundleFactory#ensure_profile_ready: loading existing profile for {config.get_doc_name()}"
                )
                return self.profile_store.load(config)
            except Exception as e:
                print(
                    f"BundleFactory#ensure_profile_ready: profile load failed, rebuilding. error={e}"
                )
                self.profile_store.clear(config)

        print(
            f"BundleFactory#ensure_profile_ready: building new profile for {config.get_doc_name()}"
        )
        profile = self.profile_builder.build(
            text=raw_text,
            document_language=document_language,
        )
        self.profile_store.save(profile, config)
        return profile

    def ensure_index_ready(self, config: StorageConfig, raw_text: str) -> FaissIndexBundle:
        """Ensure index artifacts are reusable; rebuild when needed.

Args:
    config: StorageConfig describing filesystem artifact paths.
    raw_text: Raw full document text before parsing/indexing.

Returns:
    Query-ready bundle guaranteed to match current document fingerprint."""
        doc_name = config.get_doc_name()
        has_position_metadata = self.store.has_position_metadata(config)

        if config.exists() and self.fingerprint_handler.matches(
            raw_text,
            config.get_raw_meta_path(),
        ) and has_position_metadata:
            try:
                return self.get_or_load(config, raw_text)
            except Exception as e:
                print(
                    f"BundleFactory#ensure_index_ready: load failed for {doc_name}, rebuilding. error={e}"
                )
                self.clear_artifacts(config, doc_name)
        else:
            if config.exists() and not has_position_metadata:
                print(
                    "BundleFactory#ensure_index_ready: "
                    f"legacy records schema detected for {doc_name}, rebuilding."
                )
            self.clear_artifacts(config, doc_name)

        parsed_document = self.node_provider.parse(raw_text, config)
        bundle = self.builder.build_from_parsed_document(parsed_document)

        profile = self.ensure_profile_ready(
            config=config,
            raw_text=raw_text,
            document_language=parsed_document.document_language,
        )

        bundle = bundle.set_profile(profile)

        self.store.save(bundle, config)
        self.fingerprint_handler.save(raw_text, config.get_raw_meta_path())

        self._put_cache(doc_name, bundle)
        return bundle

    def get_or_load(
        self,
        config: StorageConfig,
        raw_text: str | None = None,
    ) -> FaissIndexBundle:
        """Return cached bundle or lazily load it from persisted artifacts.

Args:
    config: StorageConfig describing filesystem artifact paths.
    raw_text: Raw full document text before parsing/indexing.

Returns:
    Cached or lazily loaded query-ready bundle for ``config``."""
        doc_name: str = config.get_doc_name()

        cached_bundle = self.bundle_cache.get(doc_name)
        if cached_bundle is not None:
            print(f"BundleFactory#get_or_load: cache hit for {doc_name}")
            self.bundle_cache.pop(doc_name)
            self.bundle_cache[doc_name] = cached_bundle
            return cached_bundle

        if not config.exists():
            raise RuntimeError(
                f"BundleFactory#get_or_load: storage for '{doc_name}' is not ready. "
                f"Call ensure_index_ready(...) first."
            )

        print(f"BundleFactory#get_or_load: lazy loading persisted bundle for {doc_name}")
        bundle = self.store.load(config)

        document_language = getattr(bundle, "document_language", "en")
        if raw_text is not None:
            profile = self.ensure_profile_ready(
                config=config,
                raw_text=raw_text,
                document_language=document_language,
            )
        else:
            if self.profile_store.exists(config):
                profile = self.profile_store.load(config)
            else:
                raise RuntimeError(
                    f"BundleFactory#get_or_load: profile for '{doc_name}' is missing. "
                    f"Provide raw_text or rebuild."
                )

        bundle = bundle.set_profile(profile)
        self._put_cache(doc_name, bundle)
        return bundle
