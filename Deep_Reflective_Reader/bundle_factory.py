from collections import OrderedDict
from typing import TYPE_CHECKING, OrderedDict as OrderedDictType

if TYPE_CHECKING:
    from document_preparation.accepted_source_guard import AcceptedSourceGuard

from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from retrieval.node_provider import NodeProvider
from config.faiss_storage_config import FaissStorageConfig
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
        """Clear retained in-memory bundles without changing persisted artifacts."""
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

    def clear_artifacts(self, config: FaissStorageConfig, doc_name: str) -> None:
        """Remove index artifacts and cached bundle, preserving the document profile.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.
    doc_name: Logical document name and artifact namespace (without extension)."""
        self.store.clear(config)
        self.fingerprint_handler.clear(config.get_raw_meta_path())
        self.invalidate(doc_name)

    def ensure_profile_ready(
        self,
        config: FaissStorageConfig,
        raw_text: str,
        document_language: str,
    ) -> DocumentProfile:
        """Reuse the persisted profile; runtime bundles cannot generate profiles.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.
    raw_text: Raw full document text before parsing/indexing.
    document_language: Primary document language code (e.g. en/zh).

Returns:
    Loaded profile for the target document."""
        if not self.profile_store.exists(config):
            raise RuntimeError("profile_not_ready: explicit initial preparation or reparse required")
        try:
            return self.profile_store.load(config)
        except Exception as error:
            raise RuntimeError("profile_unreadable: explicit reparse required") from error

    def ensure_index_ready(
        self,
        config: FaissStorageConfig,
        raw_text: str,
        *,
        source_guard: "AcceptedSourceGuard | None" = None,
    ) -> FaissIndexBundle:
        """Reuse assets, or explicitly publish a source-bound candidate.

        Raw text alone is not build authority. Callers needing a build must pass
        the accepted-source owner's live guard; initial source/profile creation
        still belongs to explicit document preparation.
        """
        doc_name = config.get_doc_name()

        def verify_source():
            from document_preparation.accepted_source_guard import AcceptedSourceGuard

            if not isinstance(source_guard, AcceptedSourceGuard):
                raise RuntimeError(
                    "index_not_ready:explicit_prepare_required:source_guard_required"
                )
            if (source_guard.snapshot.namespace != doc_name
                    or source_guard.snapshot.canonical_text != raw_text):
                raise RuntimeError(
                    "index_not_ready:explicit_prepare_required:source_guard_mismatch"
                )
            source_guard.verify_source()

        if source_guard is not None:
            verify_source()
        try:
            bundle = self.get_or_load(config, raw_text)
        except RuntimeError as error:
            if not str(error).startswith("index_not_ready:explicit_prepare_required"):
                raise
            # Only explicit source-bound repair can follow an index-readiness
            # failure. Profile errors never authorize a new candidate.
        else:
            if source_guard is not None:
                verify_source()
            return bundle

        verify_source()
        profile = self.ensure_profile_ready(
            config, raw_text, source_guard.snapshot.language
        )
        parsed_document = self.node_provider.parse(raw_text, config)
        verify_source()
        bundle = self.builder.build_from_parsed_document(parsed_document)

        def verify_publication():
            nonlocal profile
            verify_source()
            # Profile readiness may change during expensive work; never replace
            # active index assets for an already rejected runtime bundle.
            profile = self.ensure_profile_ready(
                config, raw_text, source_guard.snapshot.language
            )

        verify_publication()
        self.store.publish(
            bundle, config, text=raw_text, fingerprint=self.fingerprint_handler,
            verify_source=verify_publication,
        )
        bundle.set_profile(profile)
        self._put_cache(doc_name, bundle)
        return bundle

    def load_prepared(self, config, raw_text):
        """Read a validated publication and profile, without a rebuild fallback."""
        bundle = self.store.load_validated(config, raw_text, self.fingerprint_handler)
        profile = self.ensure_profile_ready(config, raw_text, bundle.document_language)
        return bundle.set_profile(profile)

    def get_or_load(
        self,
        config: FaissStorageConfig,
        raw_text: str | None = None,
    ) -> FaissIndexBundle:
        """Reload validated current assets and retain their bounded LRU entry.

        A logical-name hit is not evidence of a current publication. Read the
        caller's physical storage under retrieval's coherent-read boundary on
        every call, including warm lookups; no build or write fallback exists.
        """
        doc_name = config.get_doc_name()
        # A rejected lookup must not leave a retained entry masquerading as ready.
        self.invalidate(doc_name)
        if not isinstance(raw_text, str):
            raise RuntimeError(
                "index_not_ready:explicit_prepare_required:canonical_source_required"
            )
        try:
            bundle = self.store.load_validated(
                config, raw_text, self.fingerprint_handler
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise RuntimeError("index_not_ready:explicit_prepare_required") from error
        profile = self.ensure_profile_ready(config, raw_text, bundle.document_language)
        bundle.set_profile(profile)
        self._put_cache(doc_name, bundle)
        return bundle
