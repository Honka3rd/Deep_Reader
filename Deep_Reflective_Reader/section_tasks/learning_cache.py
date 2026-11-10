"""LLRN-05 cache identity and complete-result validation (no IO)."""
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import ClassVar

from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from shared.learning_results import AlignedTranslation, LearningHints, LearningProficiencyRef
from shared.learning_source import LearningDTO, SentenceResource, SentenceResourceRef, require, token


@dataclass(frozen=True, slots=True)
class TranslationKey(LearningDTO):
    resource: SentenceResourceRef
    source_language: str
    explanation_language: str
    policy_version: str
    prompt_version: str
    component: ClassVar[str] = "translation"

    def __post_init__(self):
        require(type(self.resource) is SentenceResourceRef, "learning_reference_invalid")
        for value in (self.source_language, self.explanation_language,
                      self.policy_version, self.prompt_version):
            token(value)
        require(self.source_language != self.explanation_language, "learning_languages_equal")

    @property
    def cache_id(self) -> str:
        # Class discriminator and schema are explicit; revisions/models/time are not identity.
        data = {"schema": 1, "component": self.component, **self.to_dict()}
        return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")).hexdigest()

    def validate_selection(self) -> None:
        """Current registry admission; decoding old keys must remain possible."""
        codes = LanguageProfileRegistry.get_supported_language_codes()
        require(all(code != "unknown" and code in codes
                    for code in (self.source_language, self.explanation_language)),
                "learning_language_invalid")


@dataclass(frozen=True, slots=True)
class HintsKey(TranslationKey):
    proficiency: LearningProficiencyRef
    component: ClassVar[str] = "hints"

    def __post_init__(self):
        super(HintsKey, self).__post_init__()
        require(type(self.proficiency) is LearningProficiencyRef, "learning_proficiency_invalid")

    def validate_selection(self) -> None:
        super(HintsKey, self).validate_selection()
        LanguageProficiencyRegistry.resolve_selection(language_code=self.source_language,
            scheme_id=self.proficiency.scheme_id, scheme_version=self.proficiency.scheme_version,
            level_id=self.proficiency.level_id)


CacheKey = TranslationKey | HintsKey
LearningResult = AlignedTranslation | LearningHints


def validate_result_key(key: CacheKey, result: LearningResult) -> None:
    require(type(key) in (TranslationKey, HintsKey), "learning_key_invalid")
    require(type(result) is (LearningHints if type(key) is HintsKey else AlignedTranslation),
            "learning_component_mismatch")
    ctx = result.context
    require((ctx.resource, ctx.source_language, ctx.explanation_language,
             ctx.policy_version, ctx.provenance.prompt_version)
            == (key.resource, key.source_language, key.explanation_language,
                key.policy_version, key.prompt_version), "learning_result_key_mismatch")
    if type(key) is HintsKey:
        require(result.proficiency == key.proficiency, "learning_proficiency_mismatch")


def validate_complete_result(key: CacheKey, result: LearningResult,
                             sentences: SentenceResource, effective_text: str) -> None:
    validate_result_key(key, result)
    sentences.validate_source(key.resource.source, effective_text)
    require(sentences.reference == key.resource, "learning_reference_mismatch")
    if type(result) is LearningHints:
        result.validate_against(sentences, effective_text)
    else:
        result.validate_against(sentences)
