"""Source invalidation and identity migration; no storage or provider side effects.

Invalidation is persisted in the canonical source commit. Old learning payloads
remain historical evidence, never eligible results; all reads/writes revalidate
these bindings. No read-time cleanup and no lost command receipts.
"""
from dataclasses import fields, is_dataclass, replace
from uuid import uuid4

from document_structure import learning_file_codec as codec
from document_structure.reading_content import original_blocks, project_document, state_for
from section_tasks.learning_generation_policy import BatchPlan
from shared.learning_results import TranslatedSentence, VocabularyHint
from shared.learning_source import SentenceResource, SentenceResourceRef, require


def effective_sources(document):
    if not document.chapters:
        return {}
    blocks = state_for(document).get("blocks", {})
    return {block.block_id: (chapter.chapter_id, section.section_id, unit.unit_id, block.content)
            for chapter, section, unit, block in original_blocks(project_document(document))
            if not blocks.get(block.block_id, {}).get("deleted")}


def guard_source_transition(before, after):
    """Advance only changed block identities, including deletion and restoration.

    Existing explicit mutations already assign fresh epochs; keep those tokens.
    Legacy writes that alter source without assigning one receive a fresh token.
    Unchanged blocks retain the current epoch even if a worker omitted it.
    A deliberately new document epoch (reparse) invalidates everything instead.
    """
    if before is None:
        return after
    if before.parse_provenance.get('interaction_epoch') != after.parse_provenance.get('interaction_epoch'):
        return after  # explicit reparse; ordinary saves preserve the current epoch
    old, new = effective_sources(before), effective_sources(after)
    prior = state_for(before).get('block_content_epochs', {})
    state = dict(state_for(after))
    epochs = dict(state.get('block_content_epochs', {}))
    for ident in old.keys() | new.keys() | prior.keys():
        if old.get(ident) != new.get(ident):
            if epochs.get(ident) is None or epochs.get(ident) == prior.get(ident):
                epochs[ident] = uuid4().hex
        elif ident in prior:
            epochs[ident] = prior[ident]
        else:
            epochs.pop(ident, None)
    if epochs == state.get('block_content_epochs', {}):
        return after
    state['block_content_epochs'] = epochs
    return replace(after, parse_provenance={**after.parse_provenance, 'reading_content': state})


def rename_learning_state(state, scope, old_name, new_name, *, now_ms):
    """Rebind all versions/levels/history without resegmentation or generation.

    Text, spans, model output and provenance never change. IDs and checkpoint
    digests follow the new document binding. In-flight operations become stale;
    entries advance their CAS revision, preventing old workers from installing data.
    """
    codec.validate_state(state, scope, old_name)
    require(old_name != new_name, 'learning_rename_same_identity')
    resources = {ident: resource.with_document_id(new_name) for ident, resource in state['sentences'].items()}
    sentence_ids = {old.sentence_id: new.sentence_id for ident, resource in state['sentences'].items()
                    for old, new in zip(resource.sentences, resources[ident].sentences)}

    def rebind(value):
        if type(value) is SentenceResource:
            return resources[value.reference.resource_id]
        if type(value) is SentenceResourceRef:
            if value.resource_id in resources:
                return resources[value.resource_id].reference
            # Denied/no-content receipts can predate storage of a resource. Keep
            # their old opaque digest as historical replay protection, not a
            # newly usable resource. They cannot refer to an operation/result.
            return replace(value, source=replace(value.source, document_id=new_name))
        if type(value) in (TranslatedSentence, VocabularyHint):
            return replace(value, sentence_id=sentence_ids[value.sentence_id])
        if type(value) is BatchPlan:
            return BatchPlan(tuple(tuple(sentence_ids[sid] for sid in group) for group in value.groups))
        if type(value) is tuple:
            return tuple(rebind(v) for v in value)
        if is_dataclass(value):
            return replace(value, **{f.name: rebind(getattr(value, f.name)) for f in fields(value)})
        return value

    updated = codec.empty_state(scope, new_name)
    updated['sentences'] = {r.reference.resource_id: r for r in resources.values()}
    updated['checkpoints'] = {ident: rebind(parts) for ident, parts in state['checkpoints'].items()}
    for ident, old in state['operations'].items():
        op = rebind(old)
        op = replace(op, checkpoints=tuple(codec.payload_digest(p) for p in updated['checkpoints'].get(ident, ())))
        if op.state in ('claimed', 'running', 'partial', 'unknown'):
            op = replace(op, state='stale', failure_code='binding_changed', updated_at_ms=max(now_ms, op.updated_at_ms))
        updated['operations'][ident] = op
    for old in state['entries'].values():
        entry = rebind(old)
        entry = replace(entry, revision=entry.revision + 1,
            operation=updated['operations'].get(entry.operation.operation_id) if entry.operation else None)
        updated['entries'][entry.key.cache_id] = entry
    updated['receipts'] = {ident: rebind(receipt) for ident, receipt in state['receipts'].items()}
    codec.validate_state(updated, scope, new_name)
    return updated
