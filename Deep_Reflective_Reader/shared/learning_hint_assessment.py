"""Content-free accounting for vocabulary filtering, independent of policy/IO."""
from collections import Counter
from dataclasses import dataclass

from shared.learning_source import LearningDTO, integer, payload, records, require, typed_tuple

ISSUE_CODES = (
    'invalid_candidate', 'invalid_sentence_reference', 'sentence_unavailable',
    'invalid_difficulty', 'invalid_eligibility', 'invalid_source', 'ambiguous_source',
    'invalid_kind', 'invalid_meaning', 'wrong_meaning_language', 'duplicate_conflict',
    'duplicate', 'known_level', 'proper_name', 'nonlexical', 'uncertain_difficulty',
    'foreign_sentence', 'invalid_annotation', 'missing_annotation',
    'duplicate_annotation', 'invalid_sentence_language', 'script_mismatch',
    'optional_metadata',
)


@dataclass(frozen=True, slots=True)
class HintIssue(LearningDTO):
    code: str
    count: int

    def __post_init__(self):
        require(self.code in ISSUE_CODES, 'learning_hint_issue_invalid')
        integer(self.count, minimum=1)

    @classmethod
    def from_dict(cls, data):
        return cls(**payload(cls, data))


@dataclass(frozen=True, slots=True)
class HintAssessment(LearningDTO):
    candidate_count: int
    accepted_count: int
    filtered_count: int
    rejected_count: int
    omitted_sentences: int
    issues: tuple[HintIssue, ...]

    def __post_init__(self):
        for n in (self.candidate_count, self.accepted_count, self.filtered_count,
                  self.rejected_count, self.omitted_sentences):
            integer(n)
        require(self.candidate_count == self.accepted_count + self.filtered_count + self.rejected_count,
                'learning_hint_counts_invalid')
        typed_tuple(self.issues, HintIssue)
        require(len({i.code for i in self.issues}) == len(self.issues), 'learning_hint_issue_duplicate')
        require(not (self.rejected_count or self.omitted_sentences) or bool(self.issues),
                'learning_hint_issues_required')

    @classmethod
    def from_dict(cls, data):
        values = payload(cls, data)
        values['issues'] = records(values['issues'], HintIssue)
        return cls(**values)


def combine_assessments(values):
    """Legacy unassessed batches stay unknown, never invent coverage counts."""
    values = tuple(values)
    if not values or any(v is None for v in values):
        return None
    require(all(type(v) is HintAssessment for v in values), 'learning_hint_assessment_invalid')
    counts = Counter()
    for value in values:
        counts.update({issue.code: issue.count for issue in value.issues})
    return HintAssessment(*(sum(getattr(v, name) for v in values) for name in (
        'candidate_count', 'accepted_count', 'filtered_count', 'rejected_count', 'omitted_sentences')),
        tuple(HintIssue(code, count) for code, count in sorted(counts.items())))
