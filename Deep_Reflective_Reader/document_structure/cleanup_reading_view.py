"""Pure canonical mapping of the current reading overlays for cleanup input."""

from dataclasses import dataclass

from doc_loaders.extraction_identity import text_digest
from document_structure.cleanup_input import CleanupViewPart
from document_structure.ocr_cleanup_contract import CanonicalOcrSpan
from document_structure.ocr_cleanup_overlay import canonical_text, manual_source_edits
from document_structure.reading_content import ReadingContentConflict, state_for


@dataclass(frozen=True)
class CleanupReadingView:
    parts: tuple[CleanupViewPart, ...]
    protected_ranges: tuple[CanonicalOcrSpan, ...]
    prior_deletions: tuple[CanonicalOcrSpan, ...]


def cleanup_reading_view(document) -> CleanupReadingView:
    """Reuse reading_slice precedence: manual replacements override whole overlaps.

    Keep every deleted gap explicit. Replacement text has no fabricated character
    mapping and is always protected, including empty manual deletions.
    """
    text = canonical_text(document)
    try:
        state = state_for(document)
        if ((state.get("cleanup_spans") or state.get("source_edits"))
                and state.get("cleanup_source_sha256") != text_digest(text)):
            raise ValueError
        manual = manual_source_edits(document)
        deleted = state.get("cleanup_spans", [])
        for spans, is_manual in ((manual, True), (deleted, False)):
            if not isinstance(spans, (list, tuple)):
                raise ValueError
            for span in spans:
                start, end = span["char_start"], span["char_end"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(text)
                        or span["quote"] != text[start:end]
                        or is_manual and not isinstance(span["replacement"], str)):
                    raise ValueError
            cursor = 0
            for span in sorted(spans, key=lambda item: item["char_start"]):
                if span["char_start"] < cursor:
                    raise ValueError
                cursor = span["char_end"]
        active = [span for span in deleted if not any(
            span["char_start"] < edit["char_end"] and edit["char_start"] < span["char_end"]
            for edit in manual)]
        operations = [(span, True) for span in manual] + [(span, False) for span in active]
        parts, protected, gaps = [], [], []
        cursor = 0
        for span, is_manual in sorted(operations, key=lambda item: item[0]["char_start"]):
            start, end = span["char_start"], span["char_end"]
            if cursor < start:
                parts.append(CleanupViewPart(char_start=cursor, char_end=start, text=text[cursor:start]))
            interval = CanonicalOcrSpan(char_start=start, char_end=end)
            if is_manual:
                protected.append(interval)
                parts.append(CleanupViewPart(char_start=start, char_end=end,
                                             text=span["replacement"], manual=True))
            else:
                gaps.append(interval)
            cursor = end
        if cursor < len(text):
            parts.append(CleanupViewPart(char_start=cursor, char_end=len(text), text=text[cursor:]))
        return CleanupReadingView(tuple(parts), tuple(protected), tuple(gaps))
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ReadingContentConflict("cleanup_reading_mapping_invalid") from None
