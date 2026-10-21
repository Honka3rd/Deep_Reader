"""Loader-owned PDF settings identity; never a hierarchy identity."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import shutil

from doc_loaders.pdf_extraction_contract import PdfExtractionMode


@dataclass(frozen=True)
class PdfExtractionIdentity:
    source_sha256: str
    mode: str
    extraction_version: str
    settings_sha256: str
    schema_version: int = 1

    def __post_init__(self):
        PdfExtractionMode(self.mode)
        if self.schema_version != 1 or not self.extraction_version.strip():
            raise ValueError("Unsupported extraction identity version")
        for digest in (self.source_sha256, self.settings_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Extraction identity requires SHA-256 digests")

    @classmethod
    def create(cls, *, source_sha256, mode, extraction_version, settings):
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(source_sha256, PdfExtractionMode(mode).value, extraction_version,
                   hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    def to_dict(self):
        return asdict(self)

    def record(self, raw_text: str):
        return {"identity": self.to_dict(), "text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()}

    def matches(self, record, raw_text: str) -> bool:
        # Unknown/malformed provenance is readable, but is never a proven cache hit.
        return isinstance(record, dict) and record == self.record(raw_text)


def identity_for_loader(loader, *, source_sha256: str, mode) -> PdfExtractionIdentity:
    """Fingerprint effective extraction dependencies, without executing an engine."""
    from doc_loaders.pdf_document_loader import OCR_QUALITY_VERSION
    selected = PdfExtractionMode(mode)
    versions = {PdfExtractionMode.FAST: "fast-native-v1", PdfExtractionMode.OCR: "full-page-ocr-v1",
                PdfExtractionMode.AUTO: "selective-pages-v1"}
    settings = {"pypdf": version("pypdf")}
    if selected != PdfExtractionMode.FAST:
        from .ocr_review import enabled, REVIEW_POLICY
        from .chinese_ocr_strategy import identity as chinese_ocr_identity
        def binary(command):
            path = shutil.which(command)
            return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        settings.update(language=loader.ocr_language, dpi=loader.pdf_render_dpi,
                        quality=OCR_QUALITY_VERSION, engine=binary(loader.tesseract_cmd),
                        renderer=binary(loader.pdf_renderer_cmd), page_limit=loader.ocr_page_limit,
                        timeout=loader.ocr_timeout_seconds, budgets=asdict(loader.execution_limits))
        if selected == PdfExtractionMode.OCR:
            settings["review_policy"] = REVIEW_POLICY if enabled() else "strict"
        if any(lang in {"chi_sim", "chi_sim_vert", "chi_tra", "chi_tra_vert"} for lang in loader.ocr_language.split("+")):
            settings["chinese_ocr"] = chinese_ocr_identity()
    if selected == PdfExtractionMode.AUTO:
        from doc_loaders.pdf_native_visibility import VISIBILITY_VERSION
        try:
            fonttools_version = version("fonttools")
        except PackageNotFoundError:
            fonttools_version = None
        settings.update(ocr_enabled=loader.ocr_enabled, policy=asdict(loader.ocr_policy.config),
                        policy_version=loader.ocr_policy.config.policy_version,
                        native_visibility=VISIBILITY_VERSION, fonttools=fonttools_version)
    return PdfExtractionIdentity.create(source_sha256=source_sha256, mode=selected,
                                        extraction_version=versions[selected], settings=settings)
