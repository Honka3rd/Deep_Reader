"""LLRN-10: provider-injected batch planning, generation and complete assembly.

No claim, persistence, retries or lifecycle orchestration. Only the caller that
acknowledged a fresh durable dispatch may invoke generate_batch ONCE; a looked-up
running operation is not dispatch permission. LLRN-13 owns that integration.
"""
from dataclasses import dataclass
import json
import logging
from time import monotonic

from document_structure.learning_repository import BatchPayload
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.llm_provider import LLMProvider
from section_tasks.interaction_output_language import validate_output_language
from section_tasks.learning_cache import TranslationKey, validate_complete_result
from section_tasks.learning_generation_policy import BatchPlan, GenerationOperation
from section_tasks.learning_generation_capacity import LearningGenerationCapacity, FRAMING_TOKENS
from section_tasks.learning_translation_prompt import POLICY_VERSION, PROMPT_VERSION, build_prompt
from shared.learning_results import AlignedTranslation, GenerationProvenance, LearningResultContext, TranslatedSentence
from shared.learning_source import SentenceResource, integer, require, token

logger = logging.getLogger(__name__)


class LearningTranslationError(ValueError):
    def __init__(self, code: str, *, outcome_known: bool = True):
        self.code = code
        self.outcome_known = outcome_known
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PreparedTranslation:
    key: TranslationKey
    sentences: SentenceResource
    effective_text: str
    plan: BatchPlan
    capabilities: LLMModelCapabilities
    tokenizer_identity: str | None


def _prompt(key, sentences, text, group):
    by_id = {s.sentence_id: s for s in sentences.sentences}
    selected = [by_id[sid] for sid in group]
    whole = len(group) == len(sentences.sentences)
    # Oversized inputs retain a complete contiguous sentence window and its gaps.
    # No character clipping and no full oversized paragraph repeated in each call.
    context = text if whole else text[selected[0].span.start:selected[-1].span.end]
    return build_prompt(context=context, rows=[{
        "sentence_id": s.sentence_id, "text": text[s.span.start:s.span.end]
    } for s in selected], source_language=key.source_language,
        explanation_language=key.explanation_language, partitioned=not whole)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("non-finite number")


def parse_rows(response: str, group: tuple[str, ...], target: str) -> tuple[TranslatedSentence, ...]:
    try:
        if type(response) is not str:
            raise ValueError("non-string response")
        data = json.loads(response, object_pairs_hook=_unique, parse_constant=_constant)
        if type(data) is not dict or set(data) != {"status", "rows"}:
            raise ValueError("fields")
        if data["status"] == "untranslatable" and data["rows"] == []:
            raise LearningTranslationError("learning_translation_untranslatable")
        if data["status"] != "completed" or type(data["rows"]) is not list:
            raise ValueError("status")
        rows = tuple(TranslatedSentence.from_dict(row) for row in data["rows"])
        if tuple(row.sentence_id for row in rows) != group:
            raise ValueError("alignment")
        validate_output_language("\n".join(row.translated_text for row in rows), target)
        return rows
    except LearningTranslationError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise LearningTranslationError("learning_translation_invalid_output") from None


class LearningTranslationService:
    def __init__(self, provider: LLMProvider, *, provider_name: str):
        token(provider_name)
        self._provider = provider
        self._provider_name = provider_name
        self._capacity_policy = LearningGenerationCapacity(provider, LearningTranslationError, "learning_translation_")

    def _capacity(self):
        return self._capacity_policy.capabilities()

    def _fits(self, prompt, caps):
        return self._capacity_policy.fits(prompt, caps)

    @staticmethod
    def key_for(*, sentences: SentenceResource, source_language: str, explanation_language: str):
        """Pure current-version identity for saved reads; no provider/tokenizer IO."""
        require(sentences.segmentation_language == source_language, "learning_segmentation_language_mismatch")
        key = TranslationKey(sentences.reference, source_language, explanation_language, POLICY_VERSION, PROMPT_VERSION)
        key.validate_selection()
        return key

    def prepare(self, *, sentences: SentenceResource, effective_text: str,
                source_language: str, explanation_language: str) -> PreparedTranslation | None:
        sentences.validate_source(sentences.source, effective_text)
        require(sentences.segmentation_language == source_language, "learning_segmentation_language_mismatch")
        key = self.key_for(sentences=sentences, source_language=source_language,
                           explanation_language=explanation_language)
        if not sentences.sentences:
            return None  # no_content; no model/capacity call, no fabricated result
        caps = self._capacity()
        identity = getattr(self._provider, "tokenizer_identity", None)
        plan = self._capacity_policy.plan(sentences, caps,
            lambda group: _prompt(key, sentences, effective_text, group))
        logger.info("learning_translation operation=prepare outcome=ready sentences=%d batches=%d output_limit=%d",
                    len(sentences.sentences), len(plan.groups), caps.max_output_tokens)
        return PreparedTranslation(key, sentences, effective_text, plan, caps, identity)

    def _validate(self, prepared, operation):
        require(type(prepared) is PreparedTranslation and type(operation) is GenerationOperation,
                "learning_translation_request_invalid")
        require(operation.binding.key == prepared.key and operation.plan == prepared.plan,
                "learning_translation_operation_mismatch")
        require(type(prepared.key) is TranslationKey and operation.binding.active_mode in ("language_learning", "translation"),
                "learning_translation_mode_invalid")
        require(prepared.key.policy_version == POLICY_VERSION and prepared.key.prompt_version == PROMPT_VERSION,
                "learning_translation_version_mismatch")
        prepared.key.validate_selection()
        require(prepared.sentences.segmentation_language == prepared.key.source_language,
                "learning_segmentation_language_mismatch")
        prepared.sentences.validate_source(prepared.key.resource.source, prepared.effective_text)
        require(prepared.sentences.reference == prepared.key.resource, "learning_reference_mismatch")
        prepared.plan.validate_against(prepared.sentences)

    def generate_batch(self, prepared: PreparedTranslation, operation: GenerationOperation, *, now_ms: int) -> BatchPayload:
        started, outcome = monotonic(), "failed"
        try:
            self._validate(prepared, operation)
            integer(now_ms)
            require(operation.state == "running" and operation.updated_at_ms <= now_ms < operation.lease_until_ms,
                    "learning_translation_dispatch_required")
            caps = self._capacity()
            if (caps != prepared.capabilities
                    or getattr(self._provider, "tokenizer_identity", None) != prepared.tokenizer_identity):
                raise LearningTranslationError("learning_translation_capacity_changed")
            group = prepared.plan.groups[len(operation.checkpoints)]
            prompt = _prompt(prepared.key, prepared.sentences, prepared.effective_text, group)
            if not self._fits(prompt, caps):
                raise LearningTranslationError("learning_translation_capacity_changed")
            try:
                complete = getattr(self._provider, "complete_text_with_output_budget", None)
                response = (complete(prompt, caps.max_output_tokens) if callable(complete)
                            else self._provider.complete_text(prompt))
            except Exception:
                # Transport/provider exceptions do not establish whether work was
                # performed. No automatic retry; caller records/reconciles unknown.
                raise LearningTranslationError("learning_translation_provider_outcome_unknown", outcome_known=False) from None
            rows = parse_rows(response, group, prepared.key.explanation_language)
            result = BatchPayload("translation", GenerationProvenance(
                self._provider_name, caps.model_name, PROMPT_VERSION), rows=rows)
            result.validate(operation, len(operation.checkpoints), prepared.sentences, prepared.effective_text)
            outcome = "validated"
            return result
        except LearningTranslationError as error:
            outcome = error.code
            raise
        finally:
            logger.info("learning_translation operation=generate_batch outcome=%s duration_ms=%.1f",
                        outcome, (monotonic() - started) * 1000)

    def assemble(self, prepared: PreparedTranslation, operation: GenerationOperation,
                 batches: tuple[BatchPayload, ...]) -> AlignedTranslation:
        self._validate(prepared, operation)
        require(operation.state == "partial" and len(operation.checkpoints) == len(prepared.plan.groups)
                and type(batches) is tuple and len(batches) == len(prepared.plan.groups),
                "learning_translation_incomplete")
        provenance = GenerationProvenance(self._provider_name, prepared.capabilities.model_name, PROMPT_VERSION)
        for index, batch in enumerate(batches):
            require(type(batch) is BatchPayload and batch.provenance == provenance, "learning_provenance_mismatch")
            batch.validate(operation, index, prepared.sentences, prepared.effective_text)
        key = prepared.key
        result = AlignedTranslation(LearningResultContext(key.resource, key.source_language,
            key.explanation_language, key.policy_version, operation.base_result_revision + 1, provenance),
            tuple(row for batch in batches for row in batch.rows))
        validate_complete_result(key, result, prepared.sentences, prepared.effective_text)
        # Repository commit still verifies exact equality with durable checkpoints,
        # source/settings/fence/revisions and atomically replaces output + receipt.
        return result
