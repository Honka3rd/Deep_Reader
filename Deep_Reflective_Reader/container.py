from dependency_injector import containers, providers

from bundle_factory import BundleFactory
from language.document_language_detector import DocumentLanguageDetector
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore
from faiss_index_builder import FaissIndexBuilder
from faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from node_provider import NodeProvider
from openai_embedder import OpenAIEmbedder
from openai_llm_provider import OpenAILLMProvider
from prompt_assembler import PromptAssembler
from standardized.question_standardizer import QuestionStandardizer
from storage_config import StorageConfig
from llama_index.core.node_parser import SentenceSplitter
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator
from api_key_provider import APIKeyProvider
from app_DI_config import AppDIConfig

class ApplicationLookupContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration()
    config = providers.Configuration()

    api_key_provider = providers.Singleton(APIKeyProvider)

    llm_provider = providers.Singleton(
        OpenAILLMProvider,
        api_key_provider=api_key_provider,
        model=config.llm_model,
    )

    embedder = providers.Singleton(
        OpenAIEmbedder,
        api_key_provider=api_key_provider,
        model=config.embedding_model,
    )

    prompt_assembler = providers.Singleton(PromptAssembler)

    question_standardizer = providers.Singleton(
        QuestionStandardizer,
        llm_provider=llm_provider,
    )

    relevance_evaluator = providers.Singleton(
        QuestionRelevanceEvaluator,
    )

    document_language_detector = providers.Singleton(
        DocumentLanguageDetector,
        llm_provider=llm_provider,
    )

    document_profile_builder = providers.Singleton(
        DocumentProfileBuilder,
        llm_provider=llm_provider,
    )

    document_profile_store = providers.Singleton(
        DocumentProfileStore,
    )

    sentence_splitter = providers.Factory(
        SentenceSplitter,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    node_provider = providers.Factory(
        NodeProvider,
        parser=sentence_splitter,
        detector=document_language_detector,
    )

    faiss_index_builder = providers.Singleton(
        FaissIndexBuilder,
        embedder=embedder,
        llm_provider=llm_provider,
        question_standardizer=question_standardizer,
        prompt_assembler=prompt_assembler,
        relevance_evaluator=relevance_evaluator,
        batch_size=config.embedding_batch_size,
    )

    storage_config_factory = providers.Factory(
        StorageConfig,
    )

    fingerprint_handler_factory = providers.Factory(
        FingerprintHandler,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    faiss_index_store = providers.Singleton(
        FaissIndexStore,
        embedder=embedder,
        llm_provider=llm_provider,
        question_standardizer=question_standardizer,
        prompt_assembler=prompt_assembler,
        relevance_evaluator=relevance_evaluator,
    )

    bundle_factory_provider = providers.Factory(
        BundleFactory,
        builder=faiss_index_builder,
        store=faiss_index_store,
        node_provider=node_provider,
        profile_builder=document_profile_builder,
        profile_store=document_profile_store,
        cache_capacity=config.bundle_cache_capacity,
    )

    @classmethod
    def build(cls, app_config: AppDIConfig) -> "ApplicationLookupContainer":
        container = cls()
        container.config.from_dict(
            {
                "chunk_size": app_config.chunk_size,
                "chunk_overlap": app_config.chunk_overlap,
                "embedding_model": app_config.embedding_model,
                "llm_model": app_config.llm_model,
                "embedding_batch_size": app_config.embedding_batch_size,
                "bundle_cache_capacity": app_config.bundle_cache_capacity,
            }
        )
        return container




