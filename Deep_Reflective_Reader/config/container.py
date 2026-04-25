from dependency_injector import containers, providers

from bundle_provider import BundleProvider
from bundle_factory import BundleFactory
from context.document_context_builder import DocumentContextBuilder
from context.context_orchestrator import ContextOrchestrator
from context.coverage_oriented_context_builder import CoverageOrientedContextBuilder
from context.token_budget_manager import TokenBudgetManager
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.structured_document_store import StructuredDocumentStore
from language.document_language_detector import DocumentLanguageDetector
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.faiss_index_store import FaissIndexStore
from fingerprint_handler import FingerprintHandler
from retrieval.node_provider import NodeProvider
from embeddings.openai_embedder import OpenAIEmbedder
from embeddings.embedding_similarity_service import EmbeddingSimilarityService
from llm.openai_llm_provider import OpenAILLMProvider
from prompts.prompt_assembler import PromptAssembler
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.question_scope_resolver import QuestionScopeResolver
from section_tasks.chapter_quiz_task_prompt_builder import (
    ChapterQuizTaskPromptBuilder,
)
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.section_quiz_task_prompt_builder import (
    SectionQuizTaskPromptBuilder,
)
from section_tasks.section_task_context_builder import SectionTaskContextBuilder
from section_tasks.section_task_prompt_builder_factory import (
    SectionTaskPromptBuilderFactory,
)
from section_tasks.section_task_prompt_common import SectionTaskPromptCommon
from section_tasks.summary_task_prompt_builder import SummaryTaskPromptBuilder
from section_tasks.task_unit_resolver import TaskUnitResolver
from section_tasks.topic_guidance_registry import TopicGuidanceRegistry
from app.section_task_coordinator import SectionTaskCoordinator
from question.standardized.question_standardizer import QuestionStandardizer
from config.faiss_storage_config import FaissStorageConfig
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
    similarity_service = providers.Singleton(
        EmbeddingSimilarityService,
    )

    prompt_assembler = providers.Singleton(PromptAssembler)
    token_budget_manager = providers.Singleton(
        TokenBudgetManager,
        prompt_assembler=prompt_assembler,
    )
    document_context_builder = providers.Singleton(
        DocumentContextBuilder,
        token_budget_manager=token_budget_manager,
    )

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
        target_max_input_tokens=config.target_max_input_tokens,
        target_max_output_tokens=config.target_max_output_tokens,
        target_max_context_tokens=config.target_max_context_tokens,
        input_budget_utilization_ratio=config.input_budget_utilization_ratio,
        context_budget_utilization_ratio=config.context_budget_utilization_ratio,
        batch_size=config.embedding_batch_size,
    )

    storage_config_factory = providers.Factory(
        FaissStorageConfig,
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

    structured_document_builder = providers.Singleton(
        StructuredDocumentBuilder,
    )

    structured_document_store = providers.Singleton(
        StructuredDocumentStore,
    )

    bundle_provider = providers.Singleton(
        BundleProvider,
        storage_config_factory=storage_config_factory.provider,
        fingerprint_handler_factory=fingerprint_handler_factory.provider,
        bundle_factory_provider=bundle_factory_provider.provider,
        loader_factory=document_loader_factory,
    )

    document_preparation_pipeline = providers.Singleton(
        DocumentPreparationPipeline,
        loader_factory=document_loader_factory,
        language_detector=document_language_detector,
        structured_document_builder=structured_document_builder,
        structured_document_store=structured_document_store,
        node_provider=node_provider,
        faiss_index_builder=faiss_index_builder,
        faiss_index_store=faiss_index_store,
        fingerprint_handler=fingerprint_handler_factory,
        profile_builder=document_profile_builder,
        profile_store=document_profile_store,
        bundle_provider=bundle_provider,
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
        similarity_service=similarity_service,
        llm_provider=llm_provider,
        global_similarity_threshold=config.question_scope_global_similarity_threshold,
        llm_gray_zone_min_similarity=config.question_scope_llm_gray_zone_min_similarity,
        llm_gray_zone_max_similarity=config.question_scope_llm_gray_zone_max_similarity,
        llm_fallback_enabled=config.question_scope_llm_fallback_enabled,
        llm_summary_char_limit=config.question_scope_llm_summary_char_limit,
        local_anchor_similarity_threshold=config.question_scope_local_anchor_similarity_threshold,
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
        token_budget_manager=token_budget_manager,
        document_context_builder=document_context_builder,
        base_near_chunk_threshold=config.base_near_chunk_threshold,
        min_near_chunk_threshold=config.min_near_chunk_threshold,
        max_near_chunk_threshold=config.max_near_chunk_threshold,
        global_scope_min_top_k=config.global_scope_min_top_k,
        full_text_input_budget_utilization_ratio=config.full_text_input_budget_utilization_ratio,
        full_text_context_budget_utilization_ratio=config.full_text_context_budget_utilization_ratio,
    )
    section_task_context_builder = providers.Singleton(
        SectionTaskContextBuilder,
    )
    topic_guidance_registry = providers.Singleton(
        TopicGuidanceRegistry,
        embedder=embedder,
        similarity_service=similarity_service,
        semantic_match_enabled=config.section_task_topic_semantic_match_enabled,
        semantic_similarity_threshold=config.section_task_topic_semantic_similarity_threshold,
    )
    section_task_prompt_common = providers.Singleton(
        SectionTaskPromptCommon,
        topic_guidance_registry=topic_guidance_registry,
    )
    summary_task_prompt_builder = providers.Singleton(
        SummaryTaskPromptBuilder,
        common=section_task_prompt_common,
    )
    section_quiz_task_prompt_builder = providers.Singleton(
        SectionQuizTaskPromptBuilder,
        common=section_task_prompt_common,
    )
    chapter_quiz_task_prompt_builder = providers.Singleton(
        ChapterQuizTaskPromptBuilder,
        common=section_task_prompt_common,
    )
    section_task_prompt_builder_factory = providers.Singleton(
        SectionTaskPromptBuilderFactory,
        summary_builder=summary_task_prompt_builder,
        section_quiz_builder=section_quiz_task_prompt_builder,
        chapter_quiz_builder=chapter_quiz_task_prompt_builder,
    )
    task_unit_resolver = providers.Singleton(
        TaskUnitResolver,
        task_unit_min_chars=config.task_unit_min_chars,
        task_unit_max_chars=config.task_unit_max_chars,
    )

    chapter_summary_service = providers.Singleton(
        ChapterSummaryService,
        llm_provider=llm_provider,
        context_builder=section_task_context_builder,
        prompt_builder_factory=section_task_prompt_builder_factory,
        task_unit_resolver=task_unit_resolver,
    )
    chapter_quiz_service = providers.Singleton(
        ChapterQuizService,
        llm_provider=llm_provider,
        context_builder=section_task_context_builder,
        prompt_builder_factory=section_task_prompt_builder_factory,
        task_unit_resolver=task_unit_resolver,
        quiz_min_section_chars=config.quiz_min_section_chars,
    )
    section_task_coordinator = providers.Singleton(
        SectionTaskCoordinator,
        document_preparation_pipeline=document_preparation_pipeline,
        document_profile_store=document_profile_store,
        chapter_summary_service=chapter_summary_service,
        chapter_quiz_service=chapter_quiz_service,
        task_unit_resolver=task_unit_resolver,
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
                "question_scope_global_similarity_threshold": (
                    app_config.question_scope_global_similarity_threshold
                ),
                "question_scope_llm_gray_zone_min_similarity": (
                    app_config.question_scope_llm_gray_zone_min_similarity
                ),
                "question_scope_llm_gray_zone_max_similarity": (
                    app_config.question_scope_llm_gray_zone_max_similarity
                ),
                "question_scope_llm_fallback_enabled": (
                    app_config.question_scope_llm_fallback_enabled
                ),
                "question_scope_llm_summary_char_limit": (
                    app_config.question_scope_llm_summary_char_limit
                ),
                "question_scope_local_anchor_similarity_threshold": (
                    app_config.question_scope_local_anchor_similarity_threshold
                ),
                "quiz_min_section_chars": app_config.quiz_min_section_chars,
                "section_task_topic_semantic_match_enabled": (
                    app_config.section_task_topic_semantic_match_enabled
                ),
                "section_task_topic_semantic_similarity_threshold": (
                    app_config.section_task_topic_semantic_similarity_threshold
                ),
                "task_unit_min_chars": app_config.task_unit_min_chars,
                "task_unit_max_chars": app_config.task_unit_max_chars,
            }
        )
        return container
