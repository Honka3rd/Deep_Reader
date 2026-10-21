"""Explicit common-parser intent and bounded native-fragment evidence."""

from enum import StrEnum
import re

from document_structure.source_mapped_text import (
    MAX_SOURCE_CHARS, SourceRange, build_source_mapped_text,
)
from document_structure.text_normalization import _is_cjk


COMMON_POLICY_VERSION = "common-source-preserving-v3"


class TocHint(StrEnum):
    AUTO = "auto"
    PRESENT = "present"
    ABSENT = "absent"


def fragmented_prose_ranges(text: str) -> tuple[SourceRange, ...]:
    """Require dense short CJK runs plus sentence evidence, never just short lines."""
    if len(text) > MAX_SOURCE_CHARS:
        return ()
    ranges = []
    run = []

    def finish():
        if len(run) >= 8:
            content = "".join(item[2] for item in run)
            if sum(_is_cjk(char) for char in content) >= 12 and re.search(r"[。！？]", content):
                ranges.append(SourceRange(run[0][0], run[-1][1]))
        run.clear()

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        eligible = ("\f" not in line and 0 < len(stripped) <= 3 and
                    all(_is_cjk(char) or char in "，。！？；：、‘’“”「」《》" for char in stripped))
        if re.fullmatch(r"第[一二三四五六七八九十百千]+[章節节部篇]", stripped):
            eligible = False
        # A region heading at a run's beginning has no preceding word to repair.
        if not run and stripped in {"序", "前言", "后记", "後記"}:
            eligible = False
        if eligible:
            run.append((offset, offset + len(line.rstrip("\r\n")), stripped))
        else:
            finish()
        offset += len(line)
    finish()
    return tuple(ranges)


def common_analysis_view(text: str):
    return build_source_mapped_text(text, confirmed_fragment_ranges=fragmented_prose_ranges(text))
