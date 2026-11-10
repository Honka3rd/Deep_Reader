"""Assemble already validated durable batches without any provider configuration."""
from document_structure.learning_repository import BatchPayload
from section_tasks.learning_cache import HintsKey, validate_complete_result
from shared.learning_results import AlignedTranslation, LearningHints, LearningResultContext
from shared.learning_source import require
from shared.learning_hint_assessment import combine_assessments


def assemble_checkpoints(operation, sentences, effective_text, batches):
    require(operation.state == 'partial' and type(batches) is tuple and bool(batches)
            and len(batches) == len(operation.plan.groups) == len(operation.checkpoints),
            'learning_result_incomplete')
    key = operation.binding.key
    require(all(type(batch) is BatchPayload for batch in batches), 'learning_provenance_mismatch')
    provenance = batches[0].provenance
    require(provenance.prompt_version == key.prompt_version, 'learning_provenance_mismatch')
    for index, batch in enumerate(batches):
        require(type(batch) is BatchPayload and batch.provenance == provenance, 'learning_provenance_mismatch')
        batch.validate(operation, index, sentences, effective_text)
    context = LearningResultContext(key.resource, key.source_language, key.explanation_language,
        key.policy_version, operation.base_result_revision + 1, provenance)
    result = (LearningHints(context, key.proficiency, tuple(item for batch in batches for item in batch.items),
                           assessment=combine_assessments(batch.assessment for batch in batches))
              if type(key) is HintsKey else
              AlignedTranslation(context, tuple(row for batch in batches for row in batch.rows)))
    validate_complete_result(key, result, sentences, effective_text)
    return result
