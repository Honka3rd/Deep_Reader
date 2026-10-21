"""Opt-in, read-only native PDF diagnostics; outputs counts, never source text."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from unittest.mock import patch

import pypdf

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.toc_detector import TableOfContentsDetector
from language.language_code import LanguageCode
from section_tasks.artifact_validity import _semantic_content_insufficient_reason


def coverage_report(raw_text, sections):
    """Count coverage independently of parser heuristics, including lost prefixes."""
    coverage = [0] * len(raw_text)
    invalid_ranges = 0
    for section in sections:
        start, end = section.char_start, section.char_end
        if not 0 <= start <= end <= len(raw_text):
            invalid_ranges += 1
            continue
        for index in range(start, end):
            coverage[index] += 1
    return {
        "missing_nonspace_chars": sum(not char.isspace() and count == 0
                                      for char, count in zip(raw_text, coverage)),
        "overlapping_nonspace_chars": sum(not char.isspace() and count > 1
                                          for char, count in zip(raw_text, coverage)),
        "invalid_ranges": invalid_ranges,
        "first_section_start": min((s.char_start for s in sections), default=None),
    }


def inspect_text(raw_text, language):
    sections = CommonSectionSplitter().split(raw_text, language)
    toc = TableOfContentsDetector().detect(raw_text)
    lines = [line for line in raw_text.splitlines() if line.strip()]
    return {
        "raw_chars": len(raw_text),
        "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "nonempty_lines": len(lines),
        "single_character_lines": sum(len(line.strip()) == 1 for line in lines),
        "section_count": len(sections),
        "unique_title_count": len({s.title for s in sections}),
        "toc_detected": toc.detected,
        "toc_usable": toc.usable,
        "toc_reasons": toc.reasons,
        "analysis_guard_reason": _semantic_content_insufficient_reason(
            raw_text, interaction_label="analysis"),
        "coverage": coverage_report(raw_text, sections),
    }


def diagnose_pdf(path, language):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    # Snapshot extraction must not invoke a renderer, OCR, shell or persistence.
    with patch("subprocess.run", side_effect=AssertionError("unexpected subprocess")):
        result = PdfDocumentLoader(path.parent).extract(
            path.stem, mode="fast", source_bytes=source)
        report = inspect_text(result.text, language)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("source changed during diagnostics")
    return {
        "source_sha256": digest,
        "extraction_mode": "fast",
        "extraction_contract": "PdfDocumentLoader.extract/fast",
        "pages": len(result.pages),
        "page_spans": [[b.char_start, b.char_end]
                       for b in result.boundary_evidence.boundaries],
        "source_unchanged": True,
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--language", required=True,
                        choices=[code.value for code in LanguageCode
                                 if code != LanguageCode.UNKNOWN])
    args = parser.parse_args()
    print(json.dumps({"python": platform.python_version(), "pypdf": pypdf.__version__,
                      "scope": "diagnostic baseline, not repair acceptance"}))
    for path in args.pdf:
        print(json.dumps(diagnose_pdf(path, LanguageCode(args.language))))


if __name__ == "__main__":
    main()
