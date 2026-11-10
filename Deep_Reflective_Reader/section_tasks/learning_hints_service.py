"""LLRN-11: provider-injected batch planning, generation and complete assembly.

No claim, persistence, retries or lifecycle orchestration. Only the caller that
acknowledged a fresh durable dispatch may invoke generate_batch ONCE; a looked-up
running operation is not dispatch permission. LLRN-13 owns that integration.
"""
from dataclasses import dataclass
import logging
from time import monotonic

from document_structure.learning_repository import BatchPayload
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.llm_provider import LLMProvider
from section_tasks.learning_cache import HintsKey, validate_complete_result
from section_tasks.learning_generation_policy import BatchPlan, GenerationOperation
from section_tasks.learning_generation_capacity import LearningGenerationCapacity
from section_tasks.learning_hints_prompt import POLICY_VERSION, PROMPT_VERSION, build_prompt
from section_tasks.learning_hints_output import LearningHintsError, parse_hints
from shared.learning_hint_assessment import combine_assessments
from shared.learning_results import LearningHints, LearningProficiencyRef, GenerationProvenance, LearningResultContext
from shared.learning_source import SentenceResource, integer, require, token

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedHints:
    key: HintsKey
    sentences: SentenceResource
    effective_text: str
    plan: BatchPlan
    capabilities: LLMModelCapabilities
    tokenizer_identity: str | None


class LearningHintsService:
    def __init__(self, provider: LLMProvider, *, provider_name: str):
        token(provider_name)
        self._provider = provider
        self._provider_name = provider_name
        self._capacity_policy = LearningGenerationCapacity(provider, LearningHintsError, "learning_hints_")

    def _capacity(self):
        return self._capacity_policy.capabilities()

    def _fits(self, prompt, caps):
        return self._capacity_policy.fits(prompt, caps)

    @staticmethod
    def key_for(*, sentences: SentenceResource, source_language: str, explanation_language: str, proficiency: LearningProficiencyRef):
        """Pure current-version identity for saved reads; no provider/tokenizer IO."""
        require(sentences.segmentation_language == source_language, "learning_segmentation_language_mismatch")
        key = HintsKey(sentences.reference, source_language, explanation_language, POLICY_VERSION, PROMPT_VERSION, proficiency)
        key.validate_selection()
        return key

    def prepare(self, *, sentences: SentenceResource, effective_text: str,
                source_language: str, explanation_language: str,
                proficiency: LearningProficiencyRef) -> PreparedHints | None:
        sentences.validate_source(sentences.source, effective_text)
        require(sentences.segmentation_language == source_language, "learning_segmentation_language_mismatch")
        key = self.key_for(sentences=sentences, source_language=source_language,
                           explanation_language=explanation_language, proficiency=proficiency)
        if not sentences.sentences:
            return None  # no_content; no model/capacity call, no fabricated result
        caps = self._capacity()
        identity = getattr(self._provider, "tokenizer_identity", None)
        plan = self._capacity_policy.plan(sentences, caps,
            lambda group: build_prompt(key, sentences, effective_text, group))
        logger.info("learning_hints operation=prepare outcome=ready sentences=%d batches=%d output_limit=%d",
                    len(sentences.sentences), len(plan.groups), caps.max_output_tokens)
        return PreparedHints(key, sentences, effective_text, plan, caps, identity)

    def _validate(self, prepared, operation):
        require(type(prepared) is PreparedHints and type(operation) is GenerationOperation,
                "learning_hints_request_invalid")
        require(operation.binding.key == prepared.key and operation.plan == prepared.plan,
                "learning_hints_operation_mismatch")
        require(type(prepared.key) is HintsKey and operation.binding.active_mode == "language_learning",
                "learning_hints_mode_invalid")
        require(prepared.key.policy_version == POLICY_VERSION and prepared.key.prompt_version == PROMPT_VERSION,
                "learning_hints_version_mismatch")
        prepared.key.validate_selection()
        require(prepared.sentences.segmentation_language == prepared.key.source_language,
                "learning_segmentation_language_mismatch")
        prepared.sentences.validate_source(prepared.key.resource.source, prepared.effective_text)
        require(prepared.sentences.reference == prepared.key.resource, "learning_reference_mismatch")
        prepared.plan.validate_against(prepared.sentences)

    def generate_batch(self, prepared: PreparedHints, operation: GenerationOperation, *, now_ms: int) -> BatchPayload:
        started, outcome = monotonic(), "failed"
        try:
            self._validate(prepared, operation)
            integer(now_ms)
            require(operation.state == "running" and operation.updated_at_ms <= now_ms < operation.lease_until_ms,
                    "learning_hints_dispatch_required")
            caps = self._capacity()
            if (caps != prepared.capabilities
                    or getattr(self._provider, "tokenizer_identity", None) != prepared.tokenizer_identity):
                raise LearningHintsError("learning_hints_capacity_changed")
            group = prepared.plan.groups[len(operation.checkpoints)]
            prompt = build_prompt(prepared.key, prepared.sentences, prepared.effective_text, group)
            if not self._fits(prompt, caps):
                raise LearningHintsError("learning_hints_capacity_changed")
            try:
                complete = getattr(self._provider, "complete_text_with_output_budget", None)
                response = (complete(prompt, caps.max_output_tokens) if callable(complete)
                            else self._provider.complete_text(prompt))
            except Exception:
                # Transport/provider exceptions do not establish whether work was
                # performed. No automatic retry; caller records/reconciles unknown.
                raise LearningHintsError("learning_hints_provider_outcome_unknown", outcome_known=False) from None
            items, assessment = parse_hints(response, prepared.key, prepared.sentences, prepared.effective_text, group)
            logger.info('learning_hints operation=filter accepted=%d filtered=%d rejected=%d omitted_sentences=%d issues=%s',
                assessment.accepted_count, assessment.filtered_count, assessment.rejected_count,
                assessment.omitted_sentences, ','.join(f'{i.code}:{i.count}' for i in assessment.issues))
            result = BatchPayload("hints", GenerationProvenance(
                self._provider_name, caps.model_name, PROMPT_VERSION), items=items, assessment=assessment)
            result.validate(operation, len(operation.checkpoints), prepared.sentences, prepared.effective_text)
            outcome = "validated"
            return result
        except LearningHintsError as error:
            outcome = error.code
            raise
        finally:
            logger.info("learning_hints operation=generate_batch outcome=%s duration_ms=%.1f",
                        outcome, (monotonic() - started) * 1000)

    def assemble(self, prepared: PreparedHints, operation: GenerationOperation,
                 batches: tuple[BatchPayload, ...]) -> LearningHints:
        self._validate(prepared, operation)
        require(operation.state == "partial" and len(operation.checkpoints) == len(prepared.plan.groups)
                and type(batches) is tuple and len(batches) == len(prepared.plan.groups),
                "learning_hints_incomplete")
        provenance = GenerationProvenance(self._provider_name, prepared.capabilities.model_name, PROMPT_VERSION)
        for index, batch in enumerate(batches):
            require(type(batch) is BatchPayload and batch.provenance == provenance, "learning_provenance_mismatch")
            batch.validate(operation, index, prepared.sentences, prepared.effective_text)
        key = prepared.key
        result = LearningHints(LearningResultContext(key.resource, key.source_language,
            key.explanation_language, key.policy_version, operation.base_result_revision + 1, provenance),
            key.proficiency, tuple(item for batch in batches for item in batch.items),
            assessment=combine_assessments(batch.assessment for batch in batches))
        validate_complete_result(key, result, prepared.sentences, prepared.effective_text)
        # Repository commit still verifies exact equality with durable checkpoints,
        # source/settings/fence/revisions and atomically replaces output + receipt.
        return result
