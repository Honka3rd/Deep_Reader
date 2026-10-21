"""Bounded analysis-only whitespace repair with exact canonical source offsets."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import re

from document_structure.text_normalization import _is_cjk


POLICY_VERSION = "source-mapped-cjk-v1"
MAX_SOURCE_CHARS = 2_000_000
MAX_EVIDENCE_RANGES = 4096
_WHITESPACE = re.compile(r"\s+")
_LINE_END = re.compile(r"\r\n|[\r\n\u2028\u2029\f]")


@dataclass(frozen=True)
class SourceRange:
    start: int
    end: int


@dataclass(frozen=True)
class PageRange:
    page_index: int
    start: int
    end: int


@dataclass(frozen=True)
class SourceMappedText:
    source: str
    text: str
    source_offsets: tuple[int, ...]
    line_ranges: tuple[SourceRange, ...]
    page_ranges: tuple[PageRange, ...]
    confirmed_fragment_ranges: tuple[SourceRange, ...]
    removed_ranges: tuple[SourceRange, ...]
    source_text_sha256: str
    policy_version: str = POLICY_VERSION

    def source_range(self, start: int, end: int) -> SourceRange:
        """Map a nonempty analysis interval, including any removed inner whitespace."""
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text):
            raise ValueError("analysis interval must be nonempty and in bounds")
        return SourceRange(self.source_offsets[start], self.source_offsets[end - 1] + 1)

    def source_page_ranges(self, start: int, end: int) -> tuple[PageRange, ...]:
        span = self.source_range(start, end)
        return tuple(PageRange(page.page_index, max(span.start, page.start),
                               min(span.end, page.end))
                     for page in self.page_ranges
                     if page.start < span.end and page.end > span.start)


def build_source_mapped_text(
    source: str,
    *,
    confirmed_fragment_ranges: tuple[SourceRange, ...] = (),
    page_ranges: tuple[PageRange, ...] = (),
) -> SourceMappedText:
    """Repair only supplied evidence intervals; absence of evidence means identity.

    Callers own fragmentation classification. A short line alone is not evidence.
    Ranges are canonical, sorted, nonoverlapping, and confined to a page when
    pages are supplied. This function does not infer headings, language or OCR.
    """
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError("source exceeds analysis-view character budget")
    if len(confirmed_fragment_ranges) > MAX_EVIDENCE_RANGES or len(page_ranges) > MAX_EVIDENCE_RANGES:
        raise ValueError("source evidence exceeds range budget")
    fragments = tuple(confirmed_fragment_ranges)
    pages = tuple(page_ranges)
    _validate_ranges(fragments, len(source))
    _validate_ranges(pages, len(source), allow_empty=True)
    if pages:
        previous_end = 0
        previous_index = -1
        for page in pages:
            if type(page.page_index) is not int or page.page_index <= previous_index:
                raise ValueError("page indexes must be increasing nonnegative integers")
            if source[previous_end:page.start].strip():
                raise ValueError("page evidence omits source content")
            previous_end, previous_index = page.end, page.page_index
        if source[previous_end:].strip():
            raise ValueError("page evidence omits source content")
        starts = [page.start for page in pages]
        for span in fragments:
            index = bisect_right(starts, span.start) - 1
            if index < 0 or span.end > pages[index].end:
                raise ValueError("fragment evidence must stay within one source page")

    removed = []
    fragment_index = 0
    for match in _WHITESPACE.finditer(source):
        start, end = match.span()
        while fragment_index < len(fragments) and fragments[fragment_index].end <= start:
            fragment_index += 1
        if fragment_index == len(fragments):
            break
        fragment = fragments[fragment_index]
        if not fragment.start < start < end < fragment.end:
            continue
        gap = match.group()
        # Blank lines, page separators and explicit paragraph separators are hard barriers.
        if any(char not in " \r\n\u3000" for char in gap) or len(_LINE_END.findall(gap)) >= 2:
            continue
        if _is_cjk(source[start - 1]) and _is_cjk(source[end]):
            removed.append(SourceRange(start, end))

    offsets = []
    cursor = 0
    for span in removed:
        offsets.extend(range(cursor, span.start))
        cursor = span.end
    offsets.extend(range(cursor, len(source)))
    line_ranges = []
    cursor = 0
    for match in _LINE_END.finditer(source):
        line_ranges.append(SourceRange(cursor, match.end()))
        cursor = match.end()
    if cursor < len(source):
        line_ranges.append(SourceRange(cursor, len(source)))
    return SourceMappedText(
        source=source,
        text="".join(source[index] for index in offsets),
        source_offsets=tuple(offsets),
        line_ranges=tuple(line_ranges),
        page_ranges=pages,
        confirmed_fragment_ranges=fragments,
        removed_ranges=tuple(removed),
        source_text_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _validate_ranges(ranges, source_length, *, allow_empty=False):
    previous_end = 0
    for span in ranges:
        if type(span.start) is not int or type(span.end) is not int:
            raise ValueError("source offsets must be integers")
        if not previous_end <= span.start <= span.end <= source_length:
            raise ValueError("source ranges must be ordered, disjoint and in bounds")
        if span.start == span.end and not allow_empty:
            raise ValueError("fragment evidence cannot be empty")
        previous_end = span.end
