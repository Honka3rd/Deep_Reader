"""Bounded lexical admission of requested extent, not universal query parsing."""
import re

from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from question.scope_lexical_matching import comparison_text, cue_matches

# Balanced quoted mentions are not requested ranges. Apostrophes inside words
# (don't, book's) are not quote delimiters. Work only on a comparison view.
_QUOTED = re.compile(r'''"[^"]*"|“[^”]*”|«[^»]*»|「[^」]*」|『[^』]*』|‘[^’]*’|(?<!\w)'[^']*'(?!\w)''')


def requested_extent(query: str, cues: tuple[str, ...], language: LanguageCode) -> str | None:
    """Admit an unquoted, non-excluded range; evaluate each occurrence separately.

    Adjacent prefix/postfix exclusions are finite language-owned cues. A later
    positive range is not negated by an earlier excluded or quoted occurrence.
    """
    view = comparison_text(_QUOTED.sub(' ', query))
    prefixes, suffixes = LanguageProfileRegistry.get_extent_exclusions(language)
    for cue, start, end in cue_matches(view, cues):
        before = view[:start].rstrip()
        after = view[end:].lstrip()
        excluded_before = any(stop == len(before) for _, _, stop in cue_matches(before, prefixes))
        excluded_after = any(begin == 0 for _, begin, _ in cue_matches(after, suffixes))
        if not excluded_before and not excluded_after:
            return cue
    return None
