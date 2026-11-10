"""Reading self-assessment definitions, independent of parser language profiles.

Version numbers belong to Deep Reader's curated descriptors, not exam editions.
Labels/descriptions are reading guidance, not certified scores or equivalences.
Source selection and per-language rationale: module-detailed-design.md (LLRN-01).
No model, storage, network, language detection or implicit normalization occurs here.
"""

from dataclasses import dataclass
from types import MappingProxyType

from language.language_profile_registry import LanguageProfileRegistry


class ProficiencySelectionError(ValueError):
    """Invalid caller selection; code is safe to map at a future API boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ProficiencyConfigurationError(RuntimeError):
    """Incomplete/inconsistent maintained definitions; never a user default."""


@dataclass(frozen=True, slots=True)
class ProficiencyLevel:
    level_id: str
    order: int
    label: str
    reading_description: str


@dataclass(frozen=True, slots=True)
class ProficiencyScheme:
    scheme_id: str
    version: str
    label: str
    scope_note: str
    levels: tuple[ProficiencyLevel, ...]
    reference_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LanguageProficiencyDefinition:
    language_code: str
    scheme: ProficiencyScheme


@dataclass(frozen=True, slots=True)
class ResolvedReadingProficiency:
    language_code: str
    scheme: ProficiencyScheme
    level: ProficiencyLevel

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Selection identity only; source and generation keys belong elsewhere."""
        return (self.language_code, self.scheme.scheme_id, self.scheme.version,
                self.level.level_id)


def _levels(*rows: tuple[str, str, str]) -> tuple[ProficiencyLevel, ...]:
    # Explicit sequence is authoritative: JLPT N5 precedes N1, never lexical sort.
    return tuple(ProficiencyLevel(code, index, label, description)
                 for index, (code, label, description) in enumerate(rows, start=1))


_CEFR = ProficiencyScheme(
    scheme_id="cefr_reading", version="1", label="CEFR reading levels",
    scope_note="Reading self-assessment using CEFR labels and Deep Reader summaries; "
               "not an exam score or a conversion to another scheme.",
    reference_urls=("https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions",
                    "https://rm.coe.int/1680459f97"),
    levels=_levels(
        ("A1", "A1 — Beginner", "Recognises familiar words and simple signs; needs help with most connected prose."),
        ("A2", "A2 — Elementary", "Reads brief everyday messages and finds explicit facts; unfamiliar phrasing needs support."),
        ("B1", "B1 — Intermediate", "Follows straightforward accounts on familiar subjects; needs help with figurative or specialised wording."),
        ("B2", "B2 — Upper intermediate", "Reads general articles independently; less frequent idioms and subtle wording can still need explanation."),
        ("C1", "C1 — Advanced", "Follows demanding prose and implicit meaning; unusual register and specialist vocabulary may need support."),
        ("C2", "C2 — Proficient", "Interprets dense or stylistically complex writing; rare, historical or domain-specific expressions can remain unfamiliar."),
    ),
)

_TOCFL = ProficiencyScheme(
    scheme_id="tocfl_reading", version="1", label="TOCFL reading levels",
    scope_note="Self-assessed reading using TOCFL Level 1–6 labels; Deep Reader summaries, "
               "not certification, a script preference or an HSK/CEFR conversion. "
               "Pre-Level 1 bands are outside this initial selection.",
    reference_urls=("https://tocfl.edu.tw/tocfl/index.php/test/reading/list/3",),
    levels=_levels(
        ("1", "Level 1 — 入門級 / Beginner", "Recognises familiar Chinese words in simple messages; needs substantial support for connected paragraphs."),
        ("2", "Level 2 — 基礎級 / Elementary", "Finds routine details in short Chinese notices and everyday texts; needs support beyond familiar topics."),
        ("3", "Level 3 — 進階級 / Intermediate", "Follows clear Chinese accounts of familiar events; unfamiliar idioms and abstract terms need explanation."),
        ("4", "Level 4 — 高階級 / Upper intermediate", "Understands general Chinese articles and viewpoints; literary phrasing and uncommon expressions may need support."),
        ("5", "Level 5 — 流利級 / Advanced", "Follows lengthy Chinese arguments and implied meaning; unusual register and cultural expressions may need explanation."),
        ("6", "Level 6 — 精通級 / Proficient", "Interprets complex Chinese writing and nuance; rare allusions, historical usages and specialist terms may remain unfamiliar."),
    ),
)

_JLPT = ProficiencyScheme(
    scheme_id="jlpt_reading", version="1", label="JLPT reading levels",
    scope_note="Reading self-assessment using JLPT labels and Deep Reader summaries; "
               "not a passed JLPT result or a CEFR conversion.",
    reference_urls=("https://www.jlpt.jp/e/about/levelsummary.html",),
    levels=_levels(
        ("N5", "N5 — Beginner", "Reads familiar expressions in kana and basic kanji; most continuous prose needs support."),
        ("N4", "N4 — Elementary", "Follows short Japanese passages on routine topics with common vocabulary and kanji."),
        ("N3", "N3 — Intermediate", "Understands everyday Japanese texts and headline gist; harder passages need wording explained."),
        ("N2", "N2 — Upper intermediate", "Follows general Japanese articles, commentary and narratives; uncommon idioms and nuance may need help."),
        ("N1", "N1 — Advanced", "Follows abstract Japanese arguments and author intent; rare literary or specialist expressions may still need explanation."),
    ),
)

_TOPIK = ProficiencyScheme(
    scheme_id="topik_reading", version="1", label="TOPIK reading levels",
    scope_note="Reading self-assessment using TOPIK Level 1–6 labels. Descriptions are "
               "Deep Reader reading guidance, not official pass criteria, combined "
               "test scores, speaking levels or cross-exam equivalences.",
    reference_urls=("https://www.niied.go.kr/web/NIIED/contents/niiedEng/eng_topikOverview",),
    levels=_levels(
        ("1", "Level 1 — Beginner", "Recognises familiar Korean words and short everyday expressions; connected prose needs frequent help."),
        ("2", "Level 2 — Elementary", "Reads short routine Korean messages; unfamiliar vocabulary and longer sentences need explanation."),
        ("3", "Level 3 — Intermediate", "Follows Korean paragraphs about familiar situations; abstract terms and idioms often need support."),
        ("4", "Level 4 — Upper intermediate", "Reads general Korean articles with growing independence; less common phrasing may need help."),
        ("5", "Level 5 — Advanced", "Follows extended Korean discussion and formal writing; specialised vocabulary or subtle register may need explanation."),
        ("6", "Level 6 — Proficient", "Handles demanding Korean prose; rare idioms, cultural references and technical expressions may still need support."),
    ),
)

_PRODUCT = ProficiencyScheme(
    scheme_id="deep_reader_reading", version="1", label="Deep Reader reading levels",
    scope_note="Product-defined reading self-assessment, not an official language exam "
               "or an equivalence to CEFR, JLPT, TOPIK or TOCFL.",
    reference_urls=(),
    levels=_levels(
        ("beginner", "Beginner", "Recognises a few familiar written words; needs explanations for most words in a paragraph."),
        ("elementary", "Elementary", "Reads short everyday sentences with help; new vocabulary and fixed phrases often interrupt understanding."),
        ("intermediate", "Intermediate", "Follows a straightforward paragraph on familiar topics; idioms and abstract vocabulary need support."),
        ("advanced", "Advanced", "Reads extended general prose independently; unfamiliar register, figurative language and specialist terms may need help."),
        ("proficient", "Proficient", "Interprets complex arguments and literary nuance; uncommon cultural references and technical vocabulary can still require explanation."),
    ),
)

# Associations, not a second supported-language list. Eligibility always comes
# from LanguageProfileRegistry; coverage validation requires deliberate curation
# when that registry changes. No default scheme is assigned to new languages.
_SCHEMES = MappingProxyType({s.scheme_id: s for s in (_CEFR, _TOCFL, _JLPT, _TOPIK, _PRODUCT)})
_LANGUAGE_SCHEMES = MappingProxyType({
    "en": "cefr_reading", "zh": "tocfl_reading", "ja": "jlpt_reading",
    "ko": "topik_reading", "es": "cefr_reading", "pt": "cefr_reading",
    "fr": "cefr_reading", "de": "cefr_reading", "it": "cefr_reading",
    "ru": "cefr_reading", "ar": "deep_reader_reading", "hi": "deep_reader_reading",
    "tr": "cefr_reading", "nl": "cefr_reading", "pl": "cefr_reading",
    "uk": "cefr_reading", "id": "deep_reader_reading", "vi": "deep_reader_reading",
    "th": "deep_reader_reading",
})


class LanguageProficiencyRegistry:
    """Exact registry codes only; no alias, region, level or version coercion."""

    @staticmethod
    def _language_codes() -> tuple[str, ...]:
        return tuple(code for code in LanguageProfileRegistry.get_supported_language_codes()
                     if code != "unknown")

    @classmethod
    def validate_coverage(cls) -> None:
        """Fail fast on maintained-data drift; suitable for future DI readiness."""
        codes = cls._language_codes()
        if len(codes) != len(set(codes)) or set(codes) != set(_LANGUAGE_SCHEMES):
            raise ProficiencyConfigurationError("proficiency_registry_coverage_mismatch")
        for scheme_id in _LANGUAGE_SCHEMES.values():
            if scheme_id not in _SCHEMES:
                raise ProficiencyConfigurationError("proficiency_scheme_missing")
        for key, scheme in _SCHEMES.items():
            levels = scheme.levels
            strings = (scheme.scheme_id, scheme.version, scheme.label, scheme.scope_note)
            if (key != scheme.scheme_id or any(not isinstance(s, str) or not s.strip() for s in strings)
                    or not isinstance(levels, tuple) or not levels
                    or len({level.level_id for level in levels}) != len(levels)
                    or tuple(level.order for level in levels) != tuple(range(1, len(levels) + 1))
                    or any(not isinstance(s, str) or not s.strip() for level in levels
                           for s in (level.level_id, level.label, level.reading_description))):
                raise ProficiencyConfigurationError("proficiency_definition_invalid")

    @classmethod
    def get_scheme(cls, language_code: str) -> ProficiencyScheme:
        if not isinstance(language_code, str) or language_code not in cls._language_codes():
            raise ProficiencySelectionError("proficiency_language_invalid")
        scheme_id = _LANGUAGE_SCHEMES.get(language_code)
        if scheme_id is None or scheme_id not in _SCHEMES:
            raise ProficiencyConfigurationError("proficiency_scheme_missing")
        return _SCHEMES[scheme_id]

    @classmethod
    def list_definitions(cls) -> tuple[LanguageProficiencyDefinition, ...]:
        cls.validate_coverage()
        return tuple(LanguageProficiencyDefinition(code, cls.get_scheme(code))
                     for code in cls._language_codes())

    @classmethod
    def resolve_selection(cls, *, language_code: str, scheme_id: str,
                          scheme_version: str, level_id: str) -> ResolvedReadingProficiency:
        """Validate a saved/requested selection against current curated definitions.

        Versions are mandatory: an old selection needs explicit migration or
        reselection by its owner, never silent reinterpretation as a new version.
        """
        scheme = cls.get_scheme(language_code)
        if not isinstance(scheme_id, str) or scheme_id != scheme.scheme_id:
            raise ProficiencySelectionError("proficiency_scheme_mismatch")
        if not isinstance(scheme_version, str) or scheme_version != scheme.version:
            raise ProficiencySelectionError("proficiency_version_mismatch")
        if isinstance(level_id, str):
            for level in scheme.levels:
                if level.level_id == level_id:
                    return ResolvedReadingProficiency(language_code, scheme, level)
        raise ProficiencySelectionError("proficiency_level_invalid")
