"""Resolve effective learning blocks from the canonical reading projection."""
from document_structure.learning_repository import LearningBlock
from document_structure.reading_content import original_blocks, project_document, state_for
from reading_preferences.models import LanguageLearningSettings, validate_book_write
from section_tasks.learning_cache import HintsKey, TranslationKey
from section_tasks.learning_generation_policy import GenerationBinding
from shared.learning_source import LearningSourceBinding, require, text_hash


def learning_blocks(document, scope, document_id):
    require(document.document_id == document_id and bool(document.chapters), "learning_document_invalid")
    require(not document.parse_provenance.get("reading_projected"), "learning_document_projected")
    # Reject ambiguous ownership, even when requested IDs happen to be unique.
    for ids in ([c.chapter_id for c in document.chapters],
                [s.section_id for c in document.chapters for s in c.sections],
                [u.unit_id for c in document.chapters for s in c.sections for u in s.task_units]):
        require(len(ids) == len(set(ids)), "learning_hierarchy_ambiguous")
    for chapter in document.chapters:
        for section in chapter.sections:
            require(section.parent_chapter_id in (None, chapter.chapter_id), 'learning_hierarchy_ambiguous')
            for unit in section.task_units:
                require(unit.parent_section_id in (None, section.section_id), 'learning_hierarchy_ambiguous')
    projected = project_document(document)
    state = state_for(document)
    blocks = []
    for chapter, section, unit, block in original_blocks(projected):
        if state.get("blocks", {}).get(block.block_id, {}).get("deleted"):
            continue
        source = LearningSourceBinding(scope.namespace, document_id, chapter.chapter_id,
            section.section_id, unit.unit_id, block.block_id,
            document.parse_provenance.get("storage_structure_version"),
            document.parse_provenance.get("interaction_epoch"),
            state.get("block_content_epochs", {}).get(block.block_id), text_hash(block.content))
        blocks.append(LearningBlock(source, block.content))
    require(len({b.source.content_block_id for b in blocks}) == len(blocks), "learning_hierarchy_ambiguous")
    return tuple(blocks)


def current_binding(key, workspace, book, *, purpose=None):
    """Persisted preference guards, not the LLRN-09 effective-profile service.

    App must resolve existing profile/eligibility first. The adapter independently
    enforces saved mode/language/level/revisions while the preferences lock is held.
    """
    require(not book.deleted, "learning_book_deleted")
    if purpose == 'translation':
        require(type(key) is TranslationKey, 'learning_component_mismatch')
        key.validate_selection()
        return GenerationBinding(key, 0, 0, 'translation')
    learning = next((m for m in book.settings.modes if type(m) is LanguageLearningSettings), None)
    require(learning is not None and book.settings.active_mode == "language_learning",
            "learning_mode_inactive")
    validate_book_write(book.settings, book.settings, workspace.native_language, learning.language_code)
    require(key.source_language == learning.language_code
            and key.explanation_language == workspace.native_language, "learning_settings_changed")
    if type(key) is HintsKey:
        require((key.proficiency.scheme_id, key.proficiency.scheme_version, key.proficiency.level_id)
                == (learning.scheme_id, learning.scheme_version, learning.level_id), "learning_settings_changed")
    return GenerationBinding(key, workspace.revision, book.revision, book.settings.active_mode)
