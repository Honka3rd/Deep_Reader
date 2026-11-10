"""Pure compatibility projection; paragraph text is derived, never persisted twice."""
from datetime import datetime, timezone

from shared.learning_source import require


def paragraph_record(saved, effective_text):
    value, operation = saved.value, saved.operation
    result, sentences = value.view.saved, value.sentences
    require(result is not None and sentences is not None, 'learning_result_missing')
    result.validate_against(sentences)
    # Preserve paragraph boundaries while displaying one continuous translation.
    rows = result.rows
    sentences.project(effective_text)
    parts = []
    for index, row in enumerate(rows):
        if index:
            gap = effective_text[sentences.sentences[index-1].span.end:sentences.sentences[index].span.start]
            parts.append(gap if '\n' in gap else ' ')
        parts.append(row.translated_text.strip())
    translated = ''.join(parts)
    require(bool(translated), 'learning_translation_empty')
    timestamp = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()
    return dict(source_language=value.key.source_language, target_language=value.key.explanation_language,
        status='completed', translated_text=translated,
        generated_at=timestamp(operation.created_at_ms), updated_at=timestamp(operation.updated_at_ms),
        revision=result.context.result_revision, mutation_id=operation.operation_id,
        effective_content_hash=value.key.resource.source.effective_content_hash)
