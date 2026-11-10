"""Validate model proposals and resolve exact source offsets on the backend.

Sentence language labels remain semantic model evidence, not parser/profile truth.
ID coverage and exclusion are verifiable; same-script misclassification is not.
"""
import json
from collections import Counter
from shared.learning_hint_assessment import HintAssessment, HintIssue

from language.language_profile_registry import LanguageProfileRegistry
from language.language_proficiency_registry import LanguageProficiencyRegistry
from language.language_script_registry import LanguageScriptRegistry
from section_tasks.interaction_output_language import validate_output_language
from shared.learning_results import VocabularyHint
from shared.learning_source import SourceSpan, require, text_value


class LearningHintsError(ValueError):
    def __init__(self, code, *, outcome_known=True):
        self.code, self.outcome_known = code, outcome_known
        super().__init__(code)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate_key')
        result[key] = value
    return result


def _constant(_):
    raise ValueError('nonfinite_number')


def _fields(value, names):
    require(type(value) is dict and set(value) == set(names.split()), 'fields')


def _source_start(fragment, form, left, right):
    """Resolve one exact sentence-local anchor, never guess among repeated text."""
    text_value(form)
    # Unique exact text already proves the position; optional context is unnecessary.
    first = fragment.find(form)
    require(first >= 0, 'source_form')
    if fragment.find(form, first + 1) == -1:
        return first
    text_value(left, nonempty=False)
    text_value(right, nonempty=False)
    anchor = left + form + right
    start = fragment.find(anchor)
    require(start >= 0, 'source_form')
    require(fragment.find(anchor, start + 1) == -1, 'ambiguous_source_form')
    return start + len(left)


def _annotations(data, group, by_id, text, source_language, issues):
    codes = LanguageProfileRegistry.get_supported_language_codes()
    scripts = LanguageScriptRegistry()
    annotations, duplicates = {}, set()
    for row in data:
        if type(row) is not dict or type(row.get('sentence_id')) is not str or row['sentence_id'] not in group:
            issues['invalid_annotation'] += 1
            continue
        sid = row['sentence_id']
        if sid in annotations:
            duplicates.add(sid)
        annotations[sid] = row
    eligible, foreign, omitted = set(), set(), set()
    for sid in group:
        reason = None
        if sid in duplicates:
            reason = 'duplicate_annotation'
        elif sid not in annotations:
            reason = 'missing_annotation'
        else:
            languages = annotations[sid].get('language_codes')
            if (type(languages) is not list or not languages
                    or not all(code is None or (type(code) is str and code in codes and code != 'unknown') for code in languages)):
                reason = 'invalid_sentence_language'
            elif set(languages) != {source_language}:
                foreign.add(sid)
            else:
                sentence = by_id[sid]
                fragment = text[sentence.span.start:sentence.span.end]
                expected = scripts.get_entry(source_language).default_script_system.value
                detected = scripts.detect_script_system(text=fragment, language=source_language).value
                if not (expected in ('mixed', 'unknown') or detected in ('unknown', expected)
                        or (expected in ('japanese', 'korean') and detected == 'mixed')):
                    reason = 'script_mismatch'
                else:
                    eligible.add(sid)
        if reason:
            omitted.add(sid)
            issues[reason] += 1
    return eligible, foreign, omitted


def _candidate(proposal, key, by_id, group, eligible, foreign, levels, level, text, issues):
    def reject(reason):
        return 'rejected', reason, None
    def exclude(reason):
        return 'filtered', reason, None
    if type(proposal) is not dict:
        return reject('invalid_candidate')
    sid = proposal.get('sentence_id')
    if type(sid) is not str or sid not in group:
        return reject('invalid_sentence_reference')
    if sid in foreign:
        return exclude('foreign_sentence')
    if sid not in eligible:
        return reject('sentence_unavailable')
    familiar = proposal.get('familiar_by')
    if type(familiar) is not str or familiar not in (*levels, 'beyond', 'uncertain'):
        return reject('invalid_difficulty')
    if type(proposal.get('lexical_unit')) is not bool or type(proposal.get('proper_name')) is not bool:
        return reject('invalid_eligibility')
    # Selection precedes grounding: an excluded name/basic word cannot poison useful hints.
    if proposal['proper_name']:
        return exclude('proper_name')
    if not proposal['lexical_unit']:
        return exclude('nonlexical')
    if familiar == 'uncertain':
        return exclude('uncertain_difficulty')
    if familiar != 'beyond' and levels[familiar] <= level:
        return exclude('known_level')
    sentence = by_id[sid]
    form = proposal.get('source_form')
    try:
        start = sentence.span.start + _source_start(text[sentence.span.start:sentence.span.end], form,
            proposal.get('left_context', ''), proposal.get('right_context', ''))
    except (ValueError, TypeError) as error:
        return reject('ambiguous_source' if str(error) == 'ambiguous_source_form' else 'invalid_source')
    if proposal.get('kind') not in ('word', 'phrase', 'idiom'):
        return reject('invalid_kind')
    try:
        meaning = proposal.get('meaning')
        text_value(meaning)
    except (ValueError, TypeError):
        return reject('invalid_meaning')
    try:
        validate_output_language(meaning, key.explanation_language)
    except ValueError:
        return reject('wrong_meaning_language')
    metadata = []
    for name in ('lemma', 'part_of_speech'):
        value = proposal.get(name)
        if value is not None:
            try:
                text_value(value)
            except (ValueError, TypeError):
                value = None
                issues['optional_metadata'] += 1
        metadata.append(value)
    return 'accepted', None, VocabularyHint(sid, SourceSpan(start, start + len(form)), form,
        key.source_language, proposal['kind'], meaning, *metadata)


def parse_hints(response, key, sentences, text, group):
    """Filter independent proposals; only an unreadable envelope fails the batch."""
    try:
        require(type(response) is str, 'response_type')
        data = json.loads(response, object_pairs_hook=_unique, parse_constant=_constant)
        _fields(data, 'status sentences candidates')
        if data == {'status': 'unavailable', 'sentences': [], 'candidates': []}:
            raise LearningHintsError('learning_hints_unavailable')
        require(data['status'] == 'completed' and type(data['sentences']) is list
                and type(data['candidates']) is list, 'status')
        by_id = {s.sentence_id: s for s in sentences.sentences}
        selection = LanguageProficiencyRegistry.resolve_selection(language_code=key.source_language,
            scheme_id=key.proficiency.scheme_id, scheme_version=key.proficiency.scheme_version,
            level_id=key.proficiency.level_id)
        levels = {level.level_id: level.order for level in selection.scheme.levels}
        issues = Counter()
        eligible, foreign, omitted = _annotations(data['sentences'], group, by_id, text, key.source_language, issues)
        occurrences = {}
        counts = Counter()
        for proposal in data['candidates']:
            state, reason, item = _candidate(proposal, key, by_id, group, eligible, foreign,
                levels, selection.level.order, text, issues)
            if item is not None:
                occurrences.setdefault((item.sentence_id, item.span.start, item.span.end), []).append(item)
            else:
                counts[state] += 1
                issues[reason] += 1
        items = []
        for candidates in occurrences.values():
            if all(item == candidates[0] for item in candidates):
                items.append(candidates[0])
                if len(candidates) > 1:
                    counts['filtered'] += len(candidates) - 1
                    issues['duplicate'] += len(candidates) - 1
            else:
                counts['rejected'] += len(candidates)
                issues['duplicate_conflict'] += len(candidates)
        items.sort(key=lambda item: (item.span.start, item.span.end))
        assessment = HintAssessment(len(data['candidates']), len(items), counts['filtered'], counts['rejected'],
            len(omitted), tuple(HintIssue(code, count) for code, count in sorted(issues.items())))
        return tuple(items), assessment
    except LearningHintsError:
        raise
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise LearningHintsError('learning_hints_invalid_output') from None


def parse_items(response, key, sentences, text, group):
    """Compatibility helper for callers only inspecting retained items."""
    return parse_hints(response, key, sentences, text, group)[0]
