"""Format-neutral extraction provenance, independent of document hierarchy."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractionIdentity:
    source_sha256: str
    source_format: str
    extractor_id: str
    extraction_version: str
    mode: str
    settings_sha256: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported_extraction_schema")
        if not isinstance(self.source_format, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", self.source_format):
            raise ValueError("unsupported_extraction_format")
        if not isinstance(self.mode, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", self.mode):
            raise ValueError("invalid_extraction_mode")
        for value in (self.extractor_id, self.extraction_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid_extractor")
        for value in (self.source_sha256, self.settings_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("invalid_extraction_digest")

    @classmethod
    def create(cls, *, settings, **values):
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(**values, settings_sha256=text_digest(encoded))

    def to_dict(self):
        return asdict(self)

    def record(self, text):
        return {"identity": self.to_dict(), "text_sha256": text_digest(text),
                "text_length": len(text), "completion": "complete"}

    def matches(self, record, text):
        return record == self.record(text)


@dataclass(frozen=True)
class ExtractionResult:
    identity: ExtractionIdentity
    canonical_text: str

    def __post_init__(self):
        if not isinstance(self.identity, ExtractionIdentity):
            raise ValueError("invalid_extraction_identity")
        if not isinstance(self.canonical_text, str) or not self.canonical_text.strip():
            raise ValueError("empty_extraction")

    @property
    def text_sha256(self):
        return text_digest(self.canonical_text)

    @property
    def text_length(self):
        return len(self.canonical_text)

    def record(self):
        return self.identity.record(self.canonical_text)


def normalize_pdf_identity(identity):
    values = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity)
    return ExtractionIdentity(**values, source_format="pdf", extractor_id="pdf-loader")


def read_extraction_record(provenance):
    """Normalize legacy PDF records without writes or dependency on PDF engines."""
    if not isinstance(provenance, dict):
        raise ValueError("extraction_identity_unknown")
    generic = provenance.get("extraction")
    legacy = provenance.get("pdf_extraction")
    if "extraction" in provenance and not isinstance(generic, dict):
        raise ValueError("invalid_extraction_record")
    if generic is not None:
        identity = ExtractionIdentity(**generic["identity"])
        if generic.get("completion") != "complete" or type(generic.get("text_length")) is not int or generic["text_length"] <= 0:
            raise ValueError("extraction_incomplete")
        record = dict(generic)
    elif legacy is not None:
        identity = normalize_pdf_identity(legacy["identity"])
        record = {"identity": identity.to_dict(), "text_sha256": legacy["text_sha256"]}
    else:
        raise ValueError("extraction_identity_unknown")
    if not isinstance(record.get("text_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["text_sha256"]):
        raise ValueError("invalid_text_digest")
    if generic is not None and legacy is not None:
        if normalize_pdf_identity(legacy["identity"]) != identity or legacy["text_sha256"] != record["text_sha256"]:
            raise ValueError("extraction_identity_conflict")
    return identity, record


def extraction_matches(provenance, identity, text):
    try:
        actual, record = read_extraction_record(provenance)
        expected = identity if isinstance(identity, ExtractionIdentity) else normalize_pdf_identity(identity)
        return (actual == expected and record["text_sha256"] == text_digest(text)
                and record.get("text_length", len(text)) == len(text))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
