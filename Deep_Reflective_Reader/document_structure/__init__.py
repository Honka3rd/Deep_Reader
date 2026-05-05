from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_hierarchy import (
    StructuredDocumentNode,
    StructuredNodeType,
)
from document_structure.structured_hierarchy_builder import (
    DocumentHierarchyBuilder,
    build_document_hierarchy_from_sections,
)
from document_structure.document_hierarchy_index import (
    SEVERE_HIERARCHY_WARNING_PREFIXES,
    assert_chapter_hierarchy_consistency,
    build_section_index_from_chapters,
    flatten_sections_from_chapters,
    find_section_by_chapter_title_effective,
    find_section_by_id_effective,
    find_sections_by_title_effective,
    get_effective_sections,
    is_severe_hierarchy_warning,
    migrate_legacy_sections_to_chapters,
    validate_chapter_hierarchy_consistency,
    with_sections_replaced_in_hierarchy,
    with_sections_synced_across_hierarchy_and_legacy,
    with_legacy_sections_synced_from_chapters,
)
from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
    DocumentStructureLanguageRules,
    ProfileEvidencePattern,
)
from document_structure.enhanced_parse_trigger_evaluator import (
    EnhancedParseTriggerDecision,
    EnhancedParseTriggerEvaluator,
)
from document_structure.abstract_section_splitter import AbstractSectionSplitter
from document_structure.heading_normalization import (
    ChineseChapterOcrNormalizationPlugin,
    CommonHeadingTypographyNormalizationPlugin,
    HeadingNormalizationExecutor,
    HeadingNormalizationPlugin,
    HeadingNormalizationPluginFactory,
    HeadingNormalizer,
)
from document_structure.section_split_plan import (
    AnchorMatchMode,
    SectionParserMode,
    SectionSplitPlan,
    SplitBoundaryInstruction,
)
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.section_splitter_dto import (
    HeadingCandidate,
    HeadingPrecedenceResult,
    LineInfo,
)
from document_structure.section_role import SectionRole
try:
    from document_structure.llm_section_splitter import LLMSectionSplitter
except ModuleNotFoundError:
    LLMSectionSplitter = None
try:
    from document_structure.section_splitter_selector import (
        SectionSplitterMode,
        SectionSplitterSelector,
    )
except ModuleNotFoundError:
    SectionSplitterMode = None
    SectionSplitterSelector = None
try:
    from document_structure.structured_document_builder import StructuredDocumentBuilder
except ModuleNotFoundError:
    StructuredDocumentBuilder = None
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.structured_document_store import StructuredDocumentStore

__all__ = [
    "StructuredDocument",
    "StructuredSection",
    "StructuredChapter",
    "StructuredDocumentNode",
    "StructuredNodeType",
    "DocumentHierarchyBuilder",
    "build_document_hierarchy_from_sections",
    "flatten_sections_from_chapters",
    "find_section_by_id_effective",
    "find_sections_by_title_effective",
    "find_section_by_chapter_title_effective",
    "build_section_index_from_chapters",
    "get_effective_sections",
    "migrate_legacy_sections_to_chapters",
    "SEVERE_HIERARCHY_WARNING_PREFIXES",
    "is_severe_hierarchy_warning",
    "validate_chapter_hierarchy_consistency",
    "assert_chapter_hierarchy_consistency",
    "with_sections_replaced_in_hierarchy",
    "with_sections_synced_across_hierarchy_and_legacy",
    "with_legacy_sections_synced_from_chapters",
    "DocumentStructureLanguageRegistry",
    "DocumentStructureLanguageRules",
    "ProfileEvidencePattern",
    "EnhancedParseTriggerDecision",
    "EnhancedParseTriggerEvaluator",
    "AbstractSectionSplitter",
    "HeadingNormalizationPlugin",
    "HeadingNormalizationExecutor",
    "HeadingNormalizationPluginFactory",
    "HeadingNormalizer",
    "CommonHeadingTypographyNormalizationPlugin",
    "ChineseChapterOcrNormalizationPlugin",
    "CommonSectionSplitter",
    "LLMSectionSplitter",
    "SectionSplitterMode",
    "SectionSplitterSelector",
    "AnchorMatchMode",
    "SectionParserMode",
    "SplitBoundaryInstruction",
    "SectionSplitPlan",
    "LineInfo",
    "HeadingCandidate",
    "HeadingPrecedenceResult",
    "SectionRole",
    "StructuredDocumentBuilder",
    "DocumentArtifactRepository",
    "StructuredDocumentArtifactRepository",
    "StructuredDocumentStore",
]
