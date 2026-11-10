"""Typed payloads for complete aligned translations and saved vocabulary hints.

These are not API schemas, cache/admission policy or new artifact categories.
Decoding checks shape; consumers must also validate_against the authoritative
sentence resource before accepting a provider/persisted result. Registry and
proficiency eligibility remain with their domain owners.
"""

from dataclasses import dataclass
from shared.learning_hint_assessment import HintAssessment

from shared.learning_source import (
    LearningDTO, SentenceResource, SentenceResourceRef, SourceSpan,
    integer, payload, records, require, schema, text_value, token, typed_tuple,
)


@dataclass(frozen=True, slots=True)
class GenerationProvenance(LearningDTO):
    provider: str
    model: str
    prompt_version: str

    def __post_init__(self):
        for value in (self.provider, self.model, self.prompt_version):
            token(value)

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class LearningResultContext(LearningDTO):
    resource: SentenceResourceRef
    source_language: str
    explanation_language: str
    policy_version: str
    result_revision: int
    provenance: GenerationProvenance

    def __post_init__(self):
        require(type(self.resource) is SentenceResourceRef, "learning_reference_invalid")
        for value in (self.source_language, self.explanation_language, self.policy_version):
            token(value)
        require(self.source_language != self.explanation_language, "learning_languages_equal")
        integer(self.result_revision, minimum=1)
        require(type(self.provenance) is GenerationProvenance, "learning_provenance_invalid")

    def validate_against(self, sentences: SentenceResource) -> None:
        require(type(sentences) is SentenceResource and self.resource == sentences.reference,
                "learning_reference_mismatch")

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["resource"] = SentenceResourceRef.from_dict(values["resource"])
        values["provenance"] = GenerationProvenance.from_dict(values["provenance"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TranslatedSentence(LearningDTO):
    sentence_id: str
    translated_text: str

    def __post_init__(self):
        token(self.sentence_id)
        text_value(self.translated_text)

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class AlignedTranslation(LearningDTO):
    context: LearningResultContext
    rows: tuple[TranslatedSentence, ...]
    schema_version: int = 1

    def __post_init__(self):
        schema(self.schema_version)
        require(type(self.context) is LearningResultContext, "learning_context_invalid")
        typed_tuple(self.rows, TranslatedSentence)
        ids = [row.sentence_id for row in self.rows]
        require(len(set(ids)) == len(ids), "learning_alignment_duplicate")

    def validate_against(self, sentences: SentenceResource) -> None:
        self.context.validate_against(sentences)
        require(tuple(row.sentence_id for row in self.rows)
                == tuple(sentence.sentence_id for sentence in sentences.sentences),
                "learning_alignment_invalid")

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["context"] = LearningResultContext.from_dict(values["context"])
        values["rows"] = records(values["rows"], TranslatedSentence)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class LearningProficiencyRef(LearningDTO):
    scheme_id: str
    scheme_version: str
    level_id: str

    def __post_init__(self):
        for value in (self.scheme_id, self.scheme_version, self.level_id):
            token(value)

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class VocabularyHint(LearningDTO):
    sentence_id: str
    span: SourceSpan
    source_form: str
    language_code: str
    kind: str
    meaning: str
    lemma: str | None = None
    part_of_speech: str | None = None

    def __post_init__(self):
        token(self.sentence_id)
        require(type(self.span) is SourceSpan, "learning_span_invalid")
        text_value(self.source_form)
        token(self.language_code)
        require(self.kind in ("word", "phrase", "idiom"), "learning_hint_kind_invalid")
        text_value(self.meaning)
        for value in (self.lemma, self.part_of_speech):
            if value is not None:
                text_value(value)

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["span"] = SourceSpan.from_dict(values["span"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class LearningHints(LearningDTO):
    context: LearningResultContext
    proficiency: LearningProficiencyRef
    items: tuple[VocabularyHint, ...]
    schema_version: int = 1
    assessment: HintAssessment | None = None

    def __post_init__(self):
        schema(self.schema_version)
        require(type(self.context) is LearningResultContext, "learning_context_invalid")
        require(type(self.proficiency) is LearningProficiencyRef, "learning_proficiency_invalid")
        typed_tuple(self.items, VocabularyHint)
        require(self.assessment is None or type(self.assessment) is HintAssessment,
                "learning_hint_assessment_invalid")
        if self.assessment is not None:
            require(self.assessment.accepted_count == len(self.items), "learning_hint_counts_invalid")
        occurrences = set()
        for item in self.items:
            require(item.language_code == self.context.source_language, "learning_hint_language_mismatch")
            occurrence = (item.sentence_id, item.span.start, item.span.end)
            require(occurrence not in occurrences, "learning_hint_duplicate")
            occurrences.add(occurrence)

    def validate_against(self, sentences: SentenceResource, effective_text: str) -> None:
        self.context.validate_against(sentences)
        sentences.project(effective_text)
        by_id = {sentence.sentence_id: sentence for sentence in sentences.sentences}
        for item in self.items:
            sentence = by_id.get(item.sentence_id)
            require(sentence is not None, "learning_hint_sentence_invalid")
            require(sentence.span.start <= item.span.start < item.span.end <= sentence.span.end,
                    "learning_hint_span_invalid")
            require(effective_text[item.span.start:item.span.end] == item.source_form,
                    "learning_hint_source_mismatch")

    @classmethod
    def from_dict(cls, data):
        if type(data) is dict and 'assessment' not in data:
            data = {**data, 'assessment': None}
        values = payload(cls, data)
        values["context"] = LearningResultContext.from_dict(values["context"])
        values["proficiency"] = LearningProficiencyRef.from_dict(values["proficiency"])
        values["items"] = records(values["items"], VocabularyHint)
        if values['assessment'] is not None:
            values['assessment'] = HintAssessment.from_dict(values['assessment'])
        return cls(**values)
