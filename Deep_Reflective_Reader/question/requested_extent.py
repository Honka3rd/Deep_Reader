"""Bounded lexical admission of requested extent, not universal query parsing."""
import re

from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from question.scope_lexical_matching import comparison_text, cue_matches

# Quotation can mark either a mention or an explicit request target. Apostrophes
# inside words are not quote delimiters. Work only on a comparison view.
_QUOTED = re.compile(r'''"[^"]*"|“[^”]*”|«[^»]*»|「[^」]*」|『[^』]*』|‘[^’]*’|(?<!\w)'[^']*'(?!\w)''')


def _excluded(view: str, start: int, end: int, prefixes: tuple[str, ...], suffixes: tuple[str, ...]) -> bool:
    before, after = view[:start].rstrip(), view[end:].lstrip()
    return (any(stop == len(before) for _, _, stop in cue_matches(before, prefixes))
            or any(begin == 0 for _, begin, _ in cue_matches(after, suffixes)))


def _quoted_target_view(query: str, cues: tuple[str, ...], language: LanguageCode) -> str:
    summary_prefixes, comparison_starts, connectors = LanguageProfileRegistry.get_quoted_extent_commands(language)
    prefixes, suffixes = LanguageProfileRegistry.get_extent_exclusions(language)

    def project(match: re.Match[str]) -> str:
        content = comparison_text(match.group()[1:-1])
        # The quoted target itself must be an exact extent, not a sentence which
        # happens to mention one. Other quoted text cannot authorize this target.
        if not any(start == 0 and end == len(content) for _, start, end in cue_matches(content, cues)):
            return ' '
        before = comparison_text(_QUOTED.sub(' ', query[:match.start()]))
        requested = before in summary_prefixes
        if any(start == 0 for _, start, _ in cue_matches(before, comparison_starts)):
            requested = any(end == len(before) and not _excluded(before, start, end, prefixes, suffixes)
                            for _, start, end in cue_matches(before, connectors))
        return ' ' + content + ' ' if requested else ' '

    return comparison_text(_QUOTED.sub(project, query))


def requested_extent(query: str, cues: tuple[str, ...], language: LanguageCode) -> str | None:
    """Admit requested ranges, retaining explicit quoted command targets.

    Adjacent exclusions remain occurrence-specific. Unrecognized quote roles are
    conservative mentions; this is finite command recognition, not full grammar.
    """
    view = _quoted_target_view(query, cues, language)
    prefixes, suffixes = LanguageProfileRegistry.get_extent_exclusions(language)
    for cue, start, end in cue_matches(view, cues):
        if not _excluded(view, start, end, prefixes, suffixes):
            return cue
    return None
