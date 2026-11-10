"""Versioned sentence-tailoring data keyed by the system language registry.

Small curated prefix abbreviations, not a second language allowlist or a claim
of complete linguistic coverage. No-space/unpunctuated text stays conservative.
"""
from dataclasses import dataclass
from types import MappingProxyType

from language.language_profile_registry import LanguageProfileRegistry

RULES_VERSION = "1"


@dataclass(frozen=True, slots=True)
class SentenceLanguageRules:
    prefix_abbreviations: frozenset[str]


# Entries suppress a period before a following token (titles, references, etc.).
# Ambiguous sentence-final abbreviations such as 'etc.' are not unconditional.
_PREFIXES = MappingProxyType({
    "en": ("mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "st.", "vs.", "e.g.", "i.e.", "fig.", "no."),
    "zh": (),
    "ja": (),
    "ko": (),
    "es": ("sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "pág.", "núm."),
    "pt": ("sr.", "sra.", "dr.", "dra.", "prof.", "pág.", "ex."),
    "fr": ("m.", "mme.", "mlle.", "dr.", "pr.", "prof.", "p.", "ex."),
    "de": ("dr.", "prof.", "hr.", "fr.", "nr.", "z.b.", "d.h.", "bzw."),
    "it": ("sig.", "sig.ra.", "dott.", "dott.ssa.", "prof.", "pag.", "n."),
    "ru": ("г.", "ул.", "д.", "стр.", "рис.", "им.", "т.е.", "т.д."),
    "ar": ("د.", "أ."),
    "hi": ("डॉ.", "प्रो."),
    "tr": ("dr.", "prof.", "sn.", "doç.", "no."),
    "nl": ("dhr.", "mevr.", "dr.", "prof.", "nr.", "bijv."),
    "pl": ("dr.", "prof.", "mgr.", "ul.", "np.", "str."),
    "uk": ("м.", "вул.", "ім.", "стор.", "рис.", "проф."),
    "id": ("dr.", "drs.", "prof.", "sdr.", "no.", "hlm."),
    "vi": ("ts.", "ths.", "pgs.", "gs.", "tp."),
    "th": ("ดร.", "นพ.", "พ.ศ.", "ค.ศ."),
})


def sentence_rules(language_code: str) -> SentenceLanguageRules:
    codes = tuple(c for c in LanguageProfileRegistry.get_supported_language_codes() if c != "unknown")
    if len(codes) != len(set(codes)) or set(codes) != set(_PREFIXES):
        raise ValueError("learning_sentence_registry_coverage_invalid")
    if type(language_code) is not str or language_code not in codes:
        raise ValueError("learning_sentence_language_invalid")
    return SentenceLanguageRules(frozenset(_PREFIXES[language_code]))
