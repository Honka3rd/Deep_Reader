"""Pure source/sentence DTOs. No source lookup, segmentation or persistence.

All offsets are block-local Unicode code points, [start, end). Source strings
are never normalized. Call validate_source/project with authoritative effective
text before using persisted spans; JSON shape alone does not prove provenance.
"""

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json


class LearningContractError(ValueError):
    """Stable, content-free contract error."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise LearningContractError(code)


def token(value) -> None:
    require(type(value) is str and bool(value) and value == value.strip()
            and value.isprintable(), "learning_token_invalid")


def text_value(value, *, nonempty=True) -> None:
    require(type(value) is str, "learning_text_invalid")
    require(not nonempty or bool(value.strip()), "learning_text_empty")
    # Reject isolated UTF-16 surrogate code units, but preserve valid Unicode.
    require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "learning_unicode_invalid")


def integer(value, *, minimum=0) -> None:
    require(type(value) is int and value >= minimum, "learning_integer_invalid")


def digest_value(value) -> None:
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "learning_hash_invalid")


def text_hash(text: str) -> str:
    text_value(text, nonempty=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class LearningDTO:
    """JSON-friendly copies; no coercion, fallback, migration or IO."""

    __slots__ = ()

    def to_dict(self) -> dict:
        # asdict makes detached nested values; JSON round-trip converts tuples to arrays.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))


def payload(cls, data) -> dict:
    require(type(data) is dict and set(data) == {f.name for f in fields(cls)},
            "learning_payload_fields_invalid")
    return dict(data)


def records(data, cls) -> tuple:
    require(type(data) is list, "learning_array_invalid")
    return tuple(cls.from_dict(item) for item in data)


def typed_tuple(value, cls) -> None:
    require(type(value) is tuple and all(type(item) is cls for item in value),
            "learning_items_invalid")


def schema(value) -> None:
    require(type(value) is int and value == 1, "learning_schema_unsupported")


@dataclass(frozen=True, slots=True)
class LearningSourceBinding(LearningDTO):
    namespace: str
    document_id: str
    chapter_id: str
    section_id: str
    task_unit_id: str
    content_block_id: str
    structure_version: int | None
    document_epoch: str | None
    content_epoch: str | None
    effective_content_hash: str
    scope_kind: str = "workspace"

    def __post_init__(self):
        require(self.scope_kind == "workspace", "learning_scope_unsupported")
        for value in (self.namespace, self.document_id, self.chapter_id, self.section_id,
                      self.task_unit_id, self.content_block_id):
            token(value)
        if self.structure_version is not None:
            integer(self.structure_version, minimum=1)
        for value in (self.document_epoch, self.content_epoch):
            if value is not None:
                token(value)
        digest_value(self.effective_content_hash)

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class SourceSpan(LearningDTO):
    start: int
    end: int

    def __post_init__(self):
        integer(self.start)
        integer(self.end)
        require(self.start < self.end, "learning_span_invalid")

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class SentenceSpan(LearningDTO):
    sentence_id: str
    span: SourceSpan

    def __post_init__(self):
        token(self.sentence_id)
        require(type(self.span) is SourceSpan, "learning_span_invalid")

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["span"] = SourceSpan.from_dict(values["span"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SourceTextPart:
    """Backend-only projection: sentence_id=None denotes a preserved whitespace gap."""
    sentence_id: str | None
    span: SourceSpan
    text: str


def _sentence_id(source, segmentation_version, segmentation_language, index, span):
    return "sentence:" + _digest(["learning-sentence-v1", source.to_dict(),
        segmentation_version, segmentation_language, index, span.to_dict()])


@dataclass(frozen=True, slots=True)
class SentenceResourceRef(LearningDTO):
    source: LearningSourceBinding
    resource_id: str

    def __post_init__(self):
        require(type(self.source) is LearningSourceBinding, "learning_source_invalid")
        digest_value(self.resource_id)

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["source"] = LearningSourceBinding.from_dict(values["source"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SentenceResource(LearningDTO):
    source: LearningSourceBinding
    segmentation_version: str
    segmentation_language: str | None
    source_length: int
    sentences: tuple[SentenceSpan, ...]
    schema_version: int = 1

    def __post_init__(self):
        schema(self.schema_version)
        require(type(self.source) is LearningSourceBinding, "learning_source_invalid")
        token(self.segmentation_version)
        if self.segmentation_language is not None:
            token(self.segmentation_language)
        integer(self.source_length)
        typed_tuple(self.sentences, SentenceSpan)
        end = 0
        for index, sentence in enumerate(self.sentences):
            require(end <= sentence.span.start < sentence.span.end <= self.source_length,
                    "learning_sentence_spans_invalid")
            require(sentence.sentence_id == _sentence_id(self.source, self.segmentation_version,
                    self.segmentation_language, index, sentence.span), "learning_sentence_id_invalid")
            end = sentence.span.end

    @classmethod
    def from_spans(cls, *, source: LearningSourceBinding, effective_text: str,
                   segmentation_version: str, segmentation_language: str | None,
                   spans: tuple[SourceSpan, ...]):
        """Assign IDs to already chosen spans; this does not choose sentence boundaries."""
        require(type(source) is LearningSourceBinding, "learning_source_invalid")
        text_value(effective_text, nonempty=False)
        typed_tuple(spans, SourceSpan)
        token(segmentation_version)
        if segmentation_language is not None:
            token(segmentation_language)
        sentences = tuple(SentenceSpan(_sentence_id(source, segmentation_version,
                          segmentation_language, i, span), span) for i, span in enumerate(spans))
        result = cls(source, segmentation_version, segmentation_language, len(effective_text), sentences)
        result.project(effective_text)
        return result

    def with_document_id(self, document_id: str):
        """Pure explicit rename primitive; retain spans/hash, reassign bound IDs.

        This does not validate rename authority or current source. The lifecycle
        owner must hold document locks and migrate all references atomically.
        """
        token(document_id)
        source = replace(self.source, document_id=document_id)
        sentences = tuple(SentenceSpan(_sentence_id(source, self.segmentation_version,
            self.segmentation_language, index, sentence.span), sentence.span)
            for index, sentence in enumerate(self.sentences))
        return replace(self, source=source, sentences=sentences)

    @property
    def reference(self) -> SentenceResourceRef:
        return SentenceResourceRef(self.source, _digest(["learning-resource-v1", self.to_dict()]))

    def validate_source(self, expected_source: LearningSourceBinding, effective_text: str) -> None:
        require(type(expected_source) is LearningSourceBinding and self.source == expected_source,
                "learning_source_changed")
        self.project(effective_text)

    def project(self, effective_text: str) -> tuple[SourceTextPart, ...]:
        """Exact source projection for rendering; never let JS slice code-point offsets."""
        require(text_hash(effective_text) == self.source.effective_content_hash
                and len(effective_text) == self.source_length, "learning_source_changed")
        parts = []
        end = 0
        for sentence in self.sentences:
            if sentence.span.start > end:
                gap = effective_text[end:sentence.span.start]
                require(gap.isspace(), "learning_uncovered_source")
                parts.append(SourceTextPart(None, SourceSpan(end, sentence.span.start), gap))
            body = effective_text[sentence.span.start:sentence.span.end]
            require(bool(body.strip()), "learning_sentence_empty")
            parts.append(SourceTextPart(sentence.sentence_id, sentence.span, body))
            end = sentence.span.end
        if end < self.source_length:
            gap = effective_text[end:]
            require(gap.isspace(), "learning_uncovered_source")
            parts.append(SourceTextPart(None, SourceSpan(end, self.source_length), gap))
        return tuple(parts)

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values["source"] = LearningSourceBinding.from_dict(values["source"])
        values["sentences"] = records(values["sentences"], SentenceSpan)
        return cls(**values)
