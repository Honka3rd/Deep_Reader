"""Deterministic, conservative learning segmentation of one resolved effective block.

This is a product rule set, NOT a full UAX #29 implementation or a language
classifier. No IO, LLM, profile mutation, hierarchy lookup or capacity guessing.
"""
import re
import unicodedata

from language.learning_sentence_rules import RULES_VERSION, SentenceLanguageRules, sentence_rules
from shared.learning_source import LearningSourceBinding, SentenceResource, SourceSpan, text_hash

SEGMENTATION_VERSION = f"reader-sentences-1/rules-{RULES_VERSION}/ucd-{unicodedata.unidata_version}"
_TERMINALS = frozenset('.!?。！？؟।॥…')
_CLOSERS = frozenset('\"\'”’»›」』》)]}）】〕〉')
_DIRECTIONAL_CLOSERS = frozenset('”’»›」』》')
_OPEN_BRACKETS = {'(': ')', '[': ']', '{': '}', '（': '）', '【': '】', '〔': '〕'}
_WEAK = frozenset(',，、:：;；؛')
_NEWLINE = r'(?:\r\n|\r(?!\n)|(?<!\r)\n|\u2028)'
_PARAGRAPH = re.compile(_NEWLINE + r'[^\S\r\n\u2028\u2029]*' + _NEWLINE + r'|\u2029')
_LIST = re.compile(r'[ \t]*(?:[-*•‣▪][ \t]+|(?:\d+[.)．、]|[A-Za-z][.)])[ \t]+|[一二三四五六七八九十]+、)')
# Python \w includes letters/digits but not all combining marks; prefixes below
# use literal matching too, so Devanagari abbreviations remain intact.
_DOTTED = re.compile(r'\w+(?:\.\w+)*\.?', re.UNICODE)
_URL = re.compile(r'''(?:https?://|www\.)[^\s<>"'“”«»。！？؟।॥]+''', re.IGNORECASE)


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _regions(text):
    """Blank lines/paragraph separators and explicit line-leading list items."""
    boundaries = {0, len(text)}
    boundaries.update(match.end() for match in _PARAGRAPH.finditer(text))
    markers = set()
    cursor = 0
    for line in text.splitlines(keepends=True):
        match = _LIST.match(line)
        if match:
            boundaries.add(cursor)
            markers.update(range(cursor, cursor + match.end()))
        cursor += len(line)
    points = sorted(boundaries)
    return tuple(zip(points, points[1:])), markers


def _protected_terminals(text, start, end, rules):
    protected = set()
    for match in _URL.finditer(text, start, end):
        body = match.group().rstrip('.!?…,:;)]}）】')
        protected.update(i for i in range(match.start(), match.start() + len(body)) if text[i] in _TERMINALS)
    # Prefix matching on the original string preserves offsets despite casefold
    # expansions. Word boundary uses Unicode letters/digits, not ASCII alone.
    for abbreviation in rules.prefix_abbreviations:
        pattern = re.compile(r'(?<!\w)' + re.escape(abbreviation), re.IGNORECASE)
        for match in pattern.finditer(text, start, end):
            if match.end() < end and (text[match.end()].isspace() or text[match.end()].isalpha()):
                protected.update(i for i in range(match.start(), match.end()) if text[i] == '.')
    for match in _DOTTED.finditer(text, start, end):
        if not match.group().endswith('.'):
            continue
        word = match.group()[:-1]
        pieces = word.split('.')
        # Initials/initialisms are ambiguous at sentence end; preserve rather than
        # manufacture fragments (J. R. Tolkien, U.S. Army).
        if all(len(p) == 1 and p.isupper() for p in pieces):
            protected.update(i for i in range(match.start(), match.end()) if text[i] == '.')
    return protected


def _cluster_end(text, cursor, end):
    """Keep runs, attached marks and adjacent closing punctuation with the sentence."""
    while cursor < end:
        char = text[cursor]
        if char in _TERMINALS or char in _CLOSERS or unicodedata.category(char).startswith('M'):
            cursor += 1
            continue
        # French-style space before closing guillemet; don't swallow a new ASCII quote.
        if char in ' \t\u00a0\u202f':
            next_index = cursor
            while next_index < end and text[next_index] in ' \t\u00a0\u202f':
                next_index += 1
            if next_index < end and text[next_index] in _DIRECTIONAL_CLOSERS:
                cursor = next_index + 1
                continue
        break
    return cursor


def _sentence_spans(text, start, end, markers, rules):
    start, end = _trim(text, start, end)
    if start == end:
        return []
    protected = _protected_terminals(text, start, end, rules)
    spans = []
    sentence_start = start
    stack = []
    cursor = start
    while cursor < end:
        char = text[cursor]
        if char in _OPEN_BRACKETS:
            stack.append(_OPEN_BRACKETS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        if char not in _TERMINALS or cursor in protected or cursor in markers:
            cursor += 1
            continue
        if char == '.' and cursor + 1 < end and text[cursor + 1].isalnum():
            # Decimals, versions, email/host/path token interiors (including Unicode digits).
            if cursor > start and text[cursor - 1].isalnum():
                cursor += 1
                continue
        stop = _cluster_end(text, cursor + 1, end)
        closing = text[cursor + 1:stop]
        for value in closing:
            if stack and value == stack[-1]:
                stack.pop()
        next_index = stop
        while next_index < end and text[next_index].isspace():
            next_index += 1
        next_char = text[next_index] if next_index < end else ''
        has_quote = any(value in _DIRECTIONAL_CLOSERS or value in '\"\'' for value in closing)
        has_closer = any(value in _CLOSERS for value in closing)
        ellipsis = char == '…' or text[cursor:stop].startswith('..')
        continuation = bool(stack) or next_char in _WEAK
        # Don't separate trailing combining/ZWJ sequences or dialogue attributions.
        continuation |= next_char == '\u200d'
        continuation |= bool(next_char and next_char.islower() and (char == '.' or ellipsis or has_closer))
        continuation |= bool((has_quote or ellipsis) and next_index == stop and next_char and next_char.isalpha()
                             and not next_char.isascii())
        if not continuation:
            spans.append(SourceSpan(sentence_start, stop))
            sentence_start = next_index
        cursor = max(stop, cursor + 1)
        if not continuation:
            cursor = next_index
    if sentence_start < end:
        spans.append(SourceSpan(sentence_start, end))
    return spans


def segment_learning_block(*, source: LearningSourceBinding, effective_text: str,
                           source_language: str) -> SentenceResource:
    """Caller owns hierarchy resolution and source epoch/revision checks at commit."""
    rules: SentenceLanguageRules = sentence_rules(source_language)
    if type(source) is not LearningSourceBinding or text_hash(effective_text) != source.effective_content_hash:
        raise ValueError("learning_sentence_source_changed")
    regions, markers = _regions(effective_text)
    spans = tuple(span for start, end in regions
                  for span in _sentence_spans(effective_text, start, end, markers, rules))
    return SentenceResource.from_spans(source=source, effective_text=effective_text,
        segmentation_version=SEGMENTATION_VERSION, segmentation_language=source_language, spans=spans)
