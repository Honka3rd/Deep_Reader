from typing import Dict

from fingerprint_handler import FingerprintHandler
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from bundle_factory import BundleFactory
from container import ApplicationLookupContainer
from app_DI_config import AppDIConfig
from storage_config import StorageConfig
from faiss_index_bundle import FaissIndexBundle
from search_metadata import SearchMetadata
from reading_session import ReadingSession

class Coordinator:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    session_store: Dict[str, ReadingSession]
    session_recent_limit: int

    def __init__(
            self,
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            embedding_model: str = "text-embedding-3-small",
            llm_model: str = "gpt-4.1-mini",
            embedding_batch_size: int = 64,
            bundle_cache_capacity: int = 3,
    ):
        self.app_config = AppDIConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            llm_model=llm_model,
            embedding_batch_size=embedding_batch_size,
            bundle_cache_capacity=bundle_cache_capacity,
        )

        self.container = ApplicationLookupContainer.build(self.app_config)
        self.loader_factory = DocumentLoaderFactory()
        self.session_store = {}
        self.session_recent_limit = 10

    def _build_runtime_objects(
        self,
        doc_name: str,
    ) -> tuple[StorageConfig, FingerprintHandler, BundleFactory]:
        config: StorageConfig = self.container.storage_config_factory(doc_name)
        fingerprint_handler: FingerprintHandler
        fingerprint_handler = (
            self.container.fingerprint_handler_factory()
        )

        bundle_factory: BundleFactory = self.container.bundle_factory_provider(
            fingerprint_handler=fingerprint_handler,
        )

        return config, fingerprint_handler, bundle_factory

    def get_bundle(self, doc_name: str) -> FaissIndexBundle:
        loader = self.loader_factory.get(doc_name)
        raw_text: str = loader.load(doc_name)

        config, _, bundle_factory = self._build_runtime_objects(doc_name)

        return bundle_factory.ensure_index_ready(
            config=config,
            raw_text=raw_text,
        )

    def get_or_create_session(self, session_id: str, doc_name: str) -> ReadingSession:
        session = self.session_store.get(session_id)
        if session is None:
            session = ReadingSession(
                session_id=session_id,
                doc_name=doc_name,
            )
            self.session_store[session_id] = session
            return session

        if session.doc_name != doc_name:
            # Reuse session id, but reset reading state for the new document.
            session.doc_name = doc_name
            session.active_chunk_index = None
            session.recent_chunk_indices.clear()
            session.recent_questions.clear()

        return session

    def get_session(self, session_id: str) -> ReadingSession | None:
        return self.session_store.get(session_id)

    @staticmethod
    def _extract_chunk_index(bundle: FaissIndexBundle, result: SearchMetadata) -> int | None:
        record = bundle.id_to_record.get(result.faiss_id)
        if record is None:
            return None

        chunk_index = record.chunk_index()
        if isinstance(chunk_index, int):
            return chunk_index
        return None

    def _update_session_after_ask(
        self,
        session: ReadingSession,
        question: str,
        bundle: FaissIndexBundle,
        results: list[SearchMetadata],
    ) -> None:
        session.recent_questions.append(question)
        if len(session.recent_questions) > self.session_recent_limit:
            session.recent_questions = session.recent_questions[-self.session_recent_limit:]

        if not results:
            return

        best_result = results[0]
        best_chunk_index = self._extract_chunk_index(bundle, best_result)
        if best_chunk_index is None:
            return

        session.active_chunk_index = best_chunk_index
        session.recent_chunk_indices.append(best_chunk_index)
        if len(session.recent_chunk_indices) > self.session_recent_limit:
            session.recent_chunk_indices = session.recent_chunk_indices[-self.session_recent_limit:]

    def ask(
        self,
        question: str,
        doc_name: str,
        top_k: int = 3,
        session_id: str | None = None,
    ) -> str:
        bundle = self.get_bundle(doc_name)
        session: ReadingSession | None = None
        session_active_chunk_index: int | None = None
        if session_id is not None:
            session = self.get_or_create_session(session_id, doc_name)
            session_active_chunk_index = session.active_chunk_index

        answer_text, results = bundle.answer_with_results(
            query=question,
            top_k=top_k,
            session_active_chunk_index=session_active_chunk_index,
            near_chunk_threshold=2,
            local_window_radius=1,
        )

        if session is not None:
            self._update_session_after_ask(
                session=session,
                question=question,
                bundle=bundle,
                results=results,
            )

        return answer_text
