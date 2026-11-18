"""Comparison-only multilingual cue matching; never rewrite the user's query."""
from collections.abc import Iterator
import re
import unicodedata

# CJK and Thai phrases need not be separated by whitespace. Hangul is word-spaced
# but a bounded set of grammatical particles can attach to a noun phrase.
_UNSPACED = re.compile(r'[\u3400-\u9fff\u3040-\u30ff\u0e00-\u0e7f]')
_HANGUL = re.compile(r'[\uac00-\ud7af]')
_KOREAN_PARTICLES = frozenset((
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '에게', '으로', '로',
    '와', '과', '도', '만', '처럼', '까지', '부터', '보다', '에는', '에서는', '으로는',
))


def comparison_text(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text.casefold()).split())


def _word_character(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in ('L', 'M', 'N') or category == 'Pc'


def _right_boundary(query: str, end: int, cue: str) -> bool:
    if end == len(query) or not _word_character(query[end]) or _UNSPACED.fullmatch(cue[-1]):
        return True
    if _HANGUL.fullmatch(cue[-1]):
        tail = end
        while tail < len(query) and _word_character(query[tail]):
            tail += 1
        return query[end:tail] in _KOREAN_PARTICLES
    return False


def cue_matches(query: str, cues: tuple[str, ...]) -> Iterator[tuple[str, int, int]]:
    """Yield configured cues and their spans in the normalized comparison view.

    Combining marks count as part of words; an ASCII-only regex boundary would
    wrongly split Hindi/Arabic and decomposed accented text. This is not a full
    tokenizer or inflection engine for all supported languages.
    """
    query = comparison_text(query)
    for cue in cues:
        normalized = comparison_text(cue)
        if not normalized:
            continue
        start = query.find(normalized)
        while start >= 0:
            end = start + len(normalized)
            left_ok = start == 0 or not _word_character(query[start-1]) or _UNSPACED.fullmatch(normalized[0])
            if left_ok and _right_boundary(query, end, normalized):
                yield cue, start, end
            start = query.find(normalized, start+1)


def contains_cue(query: str, cues: tuple[str, ...]) -> str | None:
    """Return the first configured boundary-valid cue, preserving cue priority."""
    return next((cue for cue, _, _ in cue_matches(query, cues)), None)
