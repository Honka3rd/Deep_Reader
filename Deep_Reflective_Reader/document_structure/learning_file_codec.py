"""Strict versioned learning codec shared by storage adapters; no read repair."""
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json

from document_structure.learning_repository import BatchPayload
from shared.learning_hint_assessment import HintAssessment, HintIssue, combine_assessments
from section_tasks.learning_cache import TranslationKey, HintsKey
from section_tasks.learning_generation_policy import (BatchPlan, CacheEntry, GenerationBinding,
    GenerationCommand, GenerationOperation, CommandReceipt)
from shared.learning_source import (LearningSourceBinding, SentenceResource, SentenceResourceRef,
    SourceSpan, SentenceSpan, require, token)
from shared.learning_results import (GenerationProvenance, LearningResultContext, LearningProficiencyRef,
    AlignedTranslation, TranslatedSentence, LearningHints, VocabularyHint)

_TYPES = {c.__name__: c for c in (BatchPayload, TranslationKey, HintsKey, BatchPlan, CacheEntry,
    GenerationBinding, GenerationCommand, GenerationOperation, CommandReceipt, LearningSourceBinding,
    SentenceResource, SentenceResourceRef, SourceSpan, SentenceSpan, GenerationProvenance,
    LearningResultContext, LearningProficiencyRef, AlignedTranslation, TranslatedSentence,
    LearningHints, VocabularyHint, HintAssessment, HintIssue)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encode(value):
    if is_dataclass(value) and type(value).__name__ in _TYPES and _TYPES[type(value).__name__] is type(value):
        # Absent legacy assessments must retain their original checkpoint digest.
        return {"type": type(value).__name__, "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)
            if not (type(value) in (LearningHints, BatchPayload) and f.name == 'assessment' and value.assessment is None)}}
    if type(value) is tuple:
        return {"tuple": [encode(v) for v in value]}
    require(value is None or type(value) in (str, int, bool), "learning_encoding_invalid")
    return value


def decode(value):
    if value is None or type(value) in (str, int, bool):
        return value
    require(type(value) is dict, "learning_encoding_invalid")
    if set(value) == {"tuple"}:
        require(type(value["tuple"]) is list, "learning_encoding_invalid")
        return tuple(decode(v) for v in value["tuple"])
    require(set(value) == {"type", "fields"} and type(value["type"]) is str
            and value["type"] in _TYPES, "learning_encoding_invalid")
    cls = _TYPES[value["type"]]
    names = {f.name for f in fields(cls)}
    allowed = (names, names - {'assessment'}) if cls in (LearningHints, BatchPayload) else (names,)
    require(type(value["fields"]) is dict and set(value["fields"]) in allowed,
            "learning_encoding_fields_invalid")
    return cls(**{k: decode(v) for k, v in value["fields"].items()})


def payload_digest(value):
    return sha256(canonical(encode(value)).encode("utf-8")).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "learning_duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("learning_json_constant_invalid")


def empty_state(scope, document_id):
    return dict(schema_version=1, namespace=scope.namespace, scope_kind=scope.kind,
                document_id=document_id, sentences={}, entries={}, operations={}, receipts={}, checkpoints={})


def validate_state(state, scope, document_id):
    require(type(state) is dict and set(state) == set(empty_state(scope, document_id)), "learning_state_invalid")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1, "learning_schema_unsupported")
    require((state["namespace"], state["scope_kind"], state["document_id"])
            == (scope.namespace, scope.kind, document_id), "learning_scope_mismatch")
    for name in ("sentences", "entries", "operations", "receipts", "checkpoints"):
        require(type(state[name]) is dict, "learning_state_invalid")

    def scoped(key):
        require(type(key) in (TranslationKey, HintsKey), "learning_key_invalid")
        source = key.resource.source
        require((source.namespace, source.document_id, source.scope_kind)
                == (scope.namespace, document_id, scope.kind), "learning_scope_mismatch")

    for resource_id, resource in state["sentences"].items():
        require(type(resource) is SentenceResource and resource.reference.resource_id == resource_id,
                "learning_resource_invalid")
        source = resource.source
        require((source.namespace, source.document_id, source.scope_kind)
                == (scope.namespace, document_id, scope.kind), "learning_scope_mismatch")
    completed = {}
    fences = {}
    for operation_id, op in state["operations"].items():
        require(type(op) is GenerationOperation and op.operation_id == operation_id, "learning_operation_invalid")
        scoped(op.binding.key)
        receipt = state["receipts"].get(operation_id)
        require(type(receipt) is CommandReceipt and type(receipt.command) is GenerationCommand
                and receipt.outcome == "claimed" and receipt.operation_id == operation_id
                and receipt.command.command_id == operation_id and receipt.command.binding == op.binding,
                "learning_claim_receipt_missing")
        cache_id = op.binding.key.cache_id
        entry = state["entries"].get(cache_id)
        require(type(entry) is CacheEntry and op.fence <= entry.last_fence, "learning_history_mismatch")
        used = fences.setdefault(cache_id, set())
        require(op.fence not in used, "learning_history_fence_duplicate")
        used.add(op.fence)
        if op.state == "succeeded":
            identity = (cache_id, op.result_revision)
            require(identity not in completed, "learning_history_revision_duplicate")
            completed[identity] = op
        resource = state["sentences"].get(op.binding.key.resource.resource_id)
        require(resource is not None and resource.reference == op.binding.key.resource, "learning_resource_missing")
        op.plan.validate_against(resource)
        parts = state["checkpoints"].get(operation_id, ())
        require(type(parts) is tuple and all(type(p) is BatchPayload for p in parts), "learning_checkpoints_invalid")
        require(tuple(payload_digest(p) for p in parts) == op.checkpoints, "learning_checkpoint_digest_mismatch")
        for index, part in enumerate(parts):
            key = op.binding.key
            require(part.component == key.component and part.provenance.prompt_version == key.prompt_version,
                    "learning_batch_component_mismatch")
            if part.component == "translation":
                require(tuple(row.sentence_id for row in part.rows) == op.plan.groups[index],
                        "learning_batch_alignment_invalid")
            else:
                require(all(h.sentence_id in op.plan.groups[index] for h in part.items), "learning_batch_hint_invalid")
    require(set(state["checkpoints"]) <= set(state["operations"]), "learning_operation_missing")
    for cache_id, entry in state["entries"].items():
        require(type(entry) is CacheEntry and entry.key.cache_id == cache_id, "learning_entry_invalid")
        scoped(entry.key)
        resource = state["sentences"].get(entry.key.resource.resource_id)
        require(resource is not None and resource.reference == entry.key.resource, "learning_resource_missing")
        if entry.operation is not None:
            require(state["operations"].get(entry.operation.operation_id) == entry.operation, "learning_history_mismatch")
        if entry.saved is not None:
            saved_op = completed.get((cache_id, entry.result_revision))
            require(saved_op is not None, "learning_result_receipt_missing")
            parts = state["checkpoints"].get(saved_op.operation_id, ())
            require(all(p.provenance == entry.saved.context.provenance for p in parts),
                    "learning_result_provenance_mismatch")
            expected = tuple(row for p in parts for row in (p.items if type(entry.key) is HintsKey else p.rows))
            actual = entry.saved.items if type(entry.key) is HintsKey else entry.saved.rows
            if type(entry.key) is HintsKey:
                require(entry.saved.assessment == combine_assessments(p.assessment for p in parts),
                        'learning_result_assessment_mismatch')
            require(actual == expected, "learning_checkpoint_result_mismatch")
            entry.saved.context.validate_against(resource)
            if type(entry.saved) is AlignedTranslation:
                entry.saved.validate_against(resource)
        if entry.operation is not None and entry.operation.state == "succeeded":
            require(entry.saved is not None and entry.result_revision == entry.operation.result_revision,
                    "learning_committed_result_missing")
    outcomes = {"reuse", "no_content", "coalesced", "reconcile", "retry_required", "unavailable", "denied", "claimed"}
    for command_id, receipt in state["receipts"].items():
        require(type(receipt) is CommandReceipt and type(receipt.command) is GenerationCommand
                and receipt.command.command_id == command_id and receipt.outcome in outcomes, "learning_receipt_invalid")
        scoped(receipt.command.binding.key)
        if receipt.operation_id is not None:
            op = state["operations"].get(receipt.operation_id)
            require(op is not None and op.binding.key == receipt.command.binding.key, "learning_receipt_operation_missing")
        if receipt.result_revision is not None:
            require(type(receipt.result_revision) is int and receipt.result_revision > 0, "learning_receipt_invalid")
        if receipt.admission_reason is not None:
            token(receipt.admission_reason)


def dumps(state):
    data = {k: ({ident: encode(v) for ident, v in value.items()} if type(value) is dict else value)
            for k, value in state.items()}
    return canonical(data)


def loads(raw, scope, document_id):
    data = json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant)
    require(type(data) is dict, "learning_state_invalid")
    for name in ("sentences", "entries", "operations", "receipts", "checkpoints"):
        require(type(data.get(name)) is dict, "learning_state_invalid")
        data[name] = {key: decode(value) for key, value in data[name].items()}
    validate_state(data, scope, document_id)
    return data
