from dependency_injector import containers, providers

from bundle_provider import BundleProvider
from bundle_factory import BundleFactory
from context.context_orchestrator import ContextOrchestrator
from context.coverage_oriented_context_builder import CoverageOrientedContextBuilder
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from language.document_language_detector import DocumentLanguageDetector
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from retrieval.node_provider import NodeProvider
from embeddings.openai_embedder import OpenAIEmbedder
from llm.openai_llm_provider import OpenAILLMProvider
from prompts.prompt_assembler import PromptAssembler
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.question_scope_resolver import QuestionScopeResolver
from question.standardized.question_standardizer import QuestionStandardizer
from config.storage_config import StorageConfig
from llama_index.core.node_parser import SentenceSplitter
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator
from auth.api_key_provider import APIKeyProvider
from auth.openai_api_key_provider import OpenAIAPIKeyProvider
from config.app_DI_config import AppDIConfig
from session.session_manager import SessionManager

class ApplicationLookupContainer(containers.DeclarativeContainer):
    """Dependency-injection container that wires runtime providers and factories."""
    wiring_config = containers.WiringConfiguration()
    config = providers.Configuration()

    api_key_provider = providers.Singleton(OpenAIAPIKeyProvider)

    llm_provider = providers.Singleton(
        OpenAILLMProvider,
        api_key_provider=api_key_provider,
        model=config.llm_model,
        target_max_output_tokens=config.target_max_output_tokens,
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
        target_max_input_tokens=config.target_max_input_tokens,
        target_max_output_tokens=config.target_max_output_tokens,
        target_max_context_tokens=config.target_max_context_tokens,
        input_budget_utilization_ratio=config.input_budget_utilization_ratio,
        context_budget_utilization_ratio=config.context_budget_utilization_ratio,
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
        target_max_input_tokens=config.target_max_input_tokens,
        target_max_output_tokens=config.target_max_output_tokens,
        target_max_context_tokens=config.target_max_context_tokens,
        input_budget_utilization_ratio=config.input_budget_utilization_ratio,
        context_budget_utilization_ratio=config.context_budget_utilization_ratio,
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

    document_loader_factory = providers.Singleton(
        DocumentLoaderFactory,
    )

    bundle_provider = providers.Singleton(
        BundleProvider,
        storage_config_factory=storage_config_factory.provider,
        fingerprint_handler_factory=fingerprint_handler_factory.provider,
        bundle_factory_provider=bundle_factory_provider.provider,
        loader_factory=document_loader_factory,
    )

    session_manager = providers.Singleton(
        SessionManager,
        session_recent_limit=config.session_recent_limit,
    )

    question_scope_keywords_provider = providers.Singleton(
        QuestionScopeKeywordsProvider,
    )

    question_scope_resolver = providers.Singleton(
        QuestionScopeResolver,
        keywords_provider=question_scope_keywords_provider,
        embedder=embedder,
    )

    global_coverage_context_builder = providers.Singleton(
        CoverageOrientedContextBuilder,
        nearby_chunk_distance=config.global_coverage_chunk_gap,
    )

    context_orchestrator = providers.Singleton(
        ContextOrchestrator,
        question_standardizer=question_standardizer,
        relevance_evaluator=relevance_evaluator,
        question_scope_resolver=question_scope_resolver,
        global_coverage_context_builder=global_coverage_context_builder,
        base_near_chunk_threshold=config.base_near_chunk_threshold,
        min_near_chunk_threshold=config.min_near_chunk_threshold,
        max_near_chunk_threshold=config.max_near_chunk_threshold,
        global_scope_min_top_k=config.global_scope_min_top_k,
        full_text_input_budget_utilization_ratio=config.full_text_input_budget_utilization_ratio,
        full_text_context_budget_utilization_ratio=config.full_text_context_budget_utilization_ratio,
    )

    @classmethod
    def build(cls, app_config: AppDIConfig) -> "ApplicationLookupContainer":
        """Build and configure application DI container.

Args:
    app_config: App config.

Returns:
    Container instance wired with values from ``app_config``."""
        container = cls()
        container.config.from_dict(
            {
                "chunk_size": app_config.chunk_size,
                "chunk_overlap": app_config.chunk_overlap,
                "embedding_model": app_config.embedding_model,
                "llm_model": app_config.llm_model,
                "target_max_input_tokens": app_config.target_max_input_tokens,
                "target_max_output_tokens": app_config.target_max_output_tokens,
                "target_max_context_tokens": app_config.target_max_context_tokens,
                "input_budget_utilization_ratio": app_config.input_budget_utilization_ratio,
                "context_budget_utilization_ratio": app_config.context_budget_utilization_ratio,
                "full_text_input_budget_utilization_ratio": (
                    app_config.full_text_input_budget_utilization_ratio
                ),
                "full_text_context_budget_utilization_ratio": (
                    app_config.full_text_context_budget_utilization_ratio
                ),
                "embedding_batch_size": app_config.embedding_batch_size,
                "bundle_cache_capacity": app_config.bundle_cache_capacity,
                "session_recent_limit": app_config.session_recent_limit,
                "base_near_chunk_threshold": app_config.base_near_chunk_threshold,
                "min_near_chunk_threshold": app_config.min_near_chunk_threshold,
                "max_near_chunk_threshold": app_config.max_near_chunk_threshold,
                "global_scope_min_top_k": app_config.global_scope_min_top_k,
                "global_coverage_chunk_gap": app_config.global_coverage_chunk_gap,
            }
        )
        return container
