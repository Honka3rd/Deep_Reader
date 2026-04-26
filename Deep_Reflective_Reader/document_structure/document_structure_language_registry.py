import re
from dataclasses import dataclass
from typing import Pattern

from language.language_code import LanguageCode


def _ensure_all_language_entries(
    marker_map: dict[LanguageCode, tuple[str, ...]],
) -> dict[LanguageCode, tuple[str, ...]]:
    """Ensure every non-UNKNOWN LanguageCode has one explicit marker entry."""
    resolved: dict[LanguageCode, tuple[str, ...]] = {}
    for language in LanguageCode:
        if language == LanguageCode.UNKNOWN:
            continue
        resolved[language] = marker_map.get(language, tuple())
    return resolved


def _compile_numeric_heading_pattern(
    aliases: tuple[str, ...],
    number_pattern: str,
) -> Pattern[str]:
    alias_part = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        rf"^\s*(?:{alias_part})\s+(?:{number_pattern})\b(?:[\s:：.\-–—].*)?\s*$",
        re.IGNORECASE,
    )


def _compile_single_heading_pattern(aliases: tuple[str, ...]) -> Pattern[str]:
    alias_part = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        rf"^\s*(?:{alias_part})\b(?:[\s:：.\-–—].*)?\s*$",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class DocumentStructureLanguageRules:
    """Language-scoped section heading rules for document structure parsing."""

    strong_heading_patterns: tuple[Pattern[str], ...]
    weak_heading_aliases: tuple[str, ...]
    weak_heading_signals: tuple[str, ...]


def _build_language_rules(
    *,
    numeric_aliases: tuple[str, ...],
    numeric_non_chapter_aliases: tuple[str, ...],
    chapter_aliases: tuple[str, ...],
    single_aliases: tuple[str, ...],
    non_chapter_number_pattern: str,
    chapter_number_pattern: str,
    extra_strong_patterns: tuple[Pattern[str], ...] = tuple(),
) -> DocumentStructureLanguageRules:
    """Build one language rule set using shared heading-pattern recipe."""
    strong_patterns: list[Pattern[str]] = []
    if numeric_non_chapter_aliases:
        strong_patterns.append(
            _compile_numeric_heading_pattern(
                numeric_non_chapter_aliases,
                non_chapter_number_pattern,
            )
        )
    if chapter_aliases:
        strong_patterns.append(
            _compile_numeric_heading_pattern(
                chapter_aliases,
                chapter_number_pattern,
            )
        )
    strong_patterns.extend(extra_strong_patterns)
    if single_aliases:
        strong_patterns.append(_compile_single_heading_pattern(single_aliases))

    weak_aliases = numeric_aliases + single_aliases
    return DocumentStructureLanguageRules(
        strong_heading_patterns=tuple(strong_patterns),
        weak_heading_aliases=weak_aliases,
        weak_heading_signals=weak_aliases,
    )


class DocumentStructureLanguageRegistry:
    """Registry for document-structure heading rules keyed by LanguageCode."""

    _ROMAN_OR_DIGIT_NUMBER = r"\d+|[ivxlcdm]+"
    _CJK_NUMBER = r"[0-9一二三四五六七八九十百千零〇两兩]+"
    _ARABIC_NUMBER = r"[0-9٠-٩]+"
    _DEVANAGARI_NUMBER = r"[0-9०-९]+"

    _EN_ALIASES_NUMERIC = ("chapter", "part", "section", "book", "lesson", "unit")
    _EN_ALIASES_NUMERIC_NON_CHAPTER = ("part", "section", "book", "lesson", "unit")
    _EN_ALIASES_SINGLE = ("prologue", "epilogue", "introduction", "appendix")
    _EN_NUMBER_WORD_ONES = "one|two|three|four|five|six|seven|eight|nine"
    _EN_NUMBER_WORD_BASE = (
        "one|two|three|four|five|six|seven|eight|nine|ten|"
        "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    )
    _EN_NUMBER_WORD_PATTERN = (
        rf"(?:{_EN_NUMBER_WORD_BASE})(?:[-\s](?:{_EN_NUMBER_WORD_ONES}))?"
    )
    _EN_CHAPTER_NUMBER_PATTERN = (
        rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_EN_NUMBER_WORD_PATTERN})"
    )
    _EN_RULES = _build_language_rules(
        numeric_aliases=_EN_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_EN_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("chapter",),
        single_aliases=_EN_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_EN_CHAPTER_NUMBER_PATTERN,
        extra_strong_patterns=(
            re.compile(
                r"^\s*appendix(?:\s+([a-z0-9]+|[ivxlcdm]+))?(?:[\s:：.\-–—].*)?\s*$",
                re.IGNORECASE,
            ),
        ),
    )

    _FR_ALIASES_NUMERIC = ("chapitre", "partie", "section", "livre", "leçon", "unité")
    _FR_ALIASES_NUMERIC_NON_CHAPTER = ("partie", "section", "livre", "leçon", "unité")
    _FR_ALIASES_SINGLE = ("prologue", "épilogue", "introduction", "annexe")
    _FR_NUMBER_WORD_ONES = "un|deux|trois|quatre|cinq|six|sept|huit|neuf"
    _FR_NUMBER_WORD_BASE = (
        "un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|"
        "quinze|seize|vingt|trente|quarante|cinquante|soixante|soixante-dix|"
        "quatre-vingt|quatre-vingt-dix"
    )
    _FR_NUMBER_WORD_PATTERN = (
        rf"(?:{_FR_NUMBER_WORD_BASE})(?:[-\s](?:et[-\s])?(?:{_FR_NUMBER_WORD_ONES}|"
        r"dix|onze|douze|treize|quatorze|quinze|seize))?"
    )
    _FR_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_FR_NUMBER_WORD_PATTERN})"
    _FR_RULES = _build_language_rules(
        numeric_aliases=_FR_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_FR_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("chapitre",),
        single_aliases=_FR_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_FR_CHAPTER_NUMBER_PATTERN,
        extra_strong_patterns=(
            re.compile(
                r"^\s*annexe(?:\s+([a-z0-9]+|[ivxlcdm]+))?(?:[\s:：.\-–—].*)?\s*$",
                re.IGNORECASE,
            ),
        ),
    )

    _DE_ALIASES_NUMERIC = ("kapitel", "teil", "abschnitt", "buch", "lektion", "einheit")
    _DE_ALIASES_NUMERIC_NON_CHAPTER = ("teil", "abschnitt", "buch", "lektion", "einheit")
    _DE_ALIASES_SINGLE = ("prolog", "epilog", "einleitung", "anhang")
    _DE_NUMBER_WORD_ONES = "eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun"
    _DE_NUMBER_WORD_BASE = (
        "eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|dreizehn|"
        "vierzehn|fünfzehn|sechzehn|siebzehn|achtzehn|neunzehn|zwanzig|dreißig|"
        "vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig"
    )
    _DE_NUMBER_WORD_PATTERN = (
        rf"(?:{_DE_NUMBER_WORD_BASE})(?:[-\s](?:und\s+)?(?:{_DE_NUMBER_WORD_ONES}))?"
    )
    _DE_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_DE_NUMBER_WORD_PATTERN})"
    _DE_RULES = _build_language_rules(
        numeric_aliases=_DE_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_DE_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("kapitel",),
        single_aliases=_DE_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_DE_CHAPTER_NUMBER_PATTERN,
        extra_strong_patterns=(
            re.compile(
                r"^\s*anhang(?:\s+([a-z0-9]+|[ivxlcdm]+))?(?:[\s:：.\-–—].*)?\s*$",
                re.IGNORECASE,
            ),
        ),
    )

    _ES_ALIASES_NUMERIC = ("capítulo", "parte", "sección", "libro", "lección", "unidad")
    _ES_ALIASES_NUMERIC_NON_CHAPTER = ("parte", "sección", "libro", "lección", "unidad")
    _ES_ALIASES_SINGLE = ("prólogo", "epílogo", "introducción", "apéndice")
    _ES_NUMBER_WORD_ONES = "uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve"
    _ES_NUMBER_WORD_BASE = (
        "uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|"
        "quince|dieciséis|diecisiete|dieciocho|diecinueve|veinte|treinta|cuarenta|"
        "cincuenta|sesenta|setenta|ochenta|noventa"
    )
    _ES_NUMBER_WORD_PATTERN = (
        rf"(?:{_ES_NUMBER_WORD_BASE})(?:\s+y\s+(?:{_ES_NUMBER_WORD_ONES}))?"
    )
    _ES_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_ES_NUMBER_WORD_PATTERN})"
    _ES_RULES = _build_language_rules(
        numeric_aliases=_ES_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_ES_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("capítulo",),
        single_aliases=_ES_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_ES_CHAPTER_NUMBER_PATTERN,
        extra_strong_patterns=(
            re.compile(
                r"^\s*apéndice(?:\s+([a-z0-9]+|[ivxlcdm]+))?(?:[\s:：.\-–—].*)?\s*$",
                re.IGNORECASE,
            ),
        ),
    )

    _PT_ALIASES_NUMERIC = ("capítulo", "parte", "seção", "livro")
    _PT_ALIASES_NUMERIC_NON_CHAPTER = ("parte", "seção", "livro")
    _PT_ALIASES_SINGLE = ("prólogo", "epílogo", "introdução", "apêndice")
    _PT_NUMBER_WORD_BASE = "um|dois|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|catorze|quinze|dezesseis|dezessete|dezoito|dezenove|vinte"
    _PT_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_PT_NUMBER_WORD_BASE})"
    _PT_RULES = _build_language_rules(
        numeric_aliases=_PT_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_PT_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("capítulo",),
        single_aliases=_PT_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_PT_CHAPTER_NUMBER_PATTERN,
    )

    _IT_ALIASES_NUMERIC = ("capitolo", "parte", "sezione", "libro")
    _IT_ALIASES_NUMERIC_NON_CHAPTER = ("parte", "sezione", "libro")
    _IT_ALIASES_SINGLE = ("prologo", "epilogo", "introduzione", "appendice")
    _IT_NUMBER_WORD_BASE = "uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|quattordici|quindici|sedici|diciassette|diciotto|diciannove|venti"
    _IT_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_IT_NUMBER_WORD_BASE})"
    _IT_RULES = _build_language_rules(
        numeric_aliases=_IT_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_IT_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("capitolo",),
        single_aliases=_IT_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_IT_CHAPTER_NUMBER_PATTERN,
    )

    _TR_ALIASES_NUMERIC = ("bölüm", "kısım", "kesit", "kitap")
    _TR_ALIASES_NUMERIC_NON_CHAPTER = ("kısım", "kesit", "kitap")
    _TR_ALIASES_SINGLE = ("önsöz", "sonsöz", "giriş", "ek")
    _TR_NUMBER_WORD_BASE = "bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|on bir|on iki"
    _TR_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_TR_NUMBER_WORD_BASE})"
    _TR_RULES = _build_language_rules(
        numeric_aliases=_TR_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_TR_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("bölüm",),
        single_aliases=_TR_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_TR_CHAPTER_NUMBER_PATTERN,
    )

    _NL_ALIASES_NUMERIC = ("hoofdstuk", "deel", "sectie", "boek")
    _NL_ALIASES_NUMERIC_NON_CHAPTER = ("deel", "sectie", "boek")
    _NL_ALIASES_SINGLE = ("proloog", "epiloog", "inleiding", "bijlage")
    _NL_NUMBER_WORD_BASE = "een|twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf"
    _NL_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_NL_NUMBER_WORD_BASE})"
    _NL_RULES = _build_language_rules(
        numeric_aliases=_NL_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_NL_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("hoofdstuk",),
        single_aliases=_NL_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_NL_CHAPTER_NUMBER_PATTERN,
    )

    _PL_ALIASES_NUMERIC = ("rozdział", "część", "sekcja", "księga")
    _PL_ALIASES_NUMERIC_NON_CHAPTER = ("część", "sekcja", "księga")
    _PL_ALIASES_SINGLE = ("prolog", "epilog", "wstęp", "aneks")
    _PL_NUMBER_WORD_BASE = "jeden|dwa|trzy|cztery|pięć|sześć|siedem|osiem|dziewięć|dziesięć|jedenaście|dwanaście"
    _PL_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_PL_NUMBER_WORD_BASE})"
    _PL_RULES = _build_language_rules(
        numeric_aliases=_PL_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_PL_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("rozdział",),
        single_aliases=_PL_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_PL_CHAPTER_NUMBER_PATTERN,
    )

    _ID_ALIASES_NUMERIC = ("bab", "bagian", "seksi", "buku")
    _ID_ALIASES_NUMERIC_NON_CHAPTER = ("bagian", "seksi", "buku")
    _ID_ALIASES_SINGLE = ("prolog", "epilog", "pendahuluan", "lampiran")
    _ID_NUMBER_WORD_BASE = "satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|dua belas"
    _ID_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_ID_NUMBER_WORD_BASE})"
    _ID_RULES = _build_language_rules(
        numeric_aliases=_ID_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_ID_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("bab",),
        single_aliases=_ID_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_ID_CHAPTER_NUMBER_PATTERN,
    )

    _VI_ALIASES_NUMERIC = ("chương", "phần", "mục", "quyển")
    _VI_ALIASES_NUMERIC_NON_CHAPTER = ("phần", "mục", "quyển")
    _VI_ALIASES_SINGLE = ("lời mở đầu", "kết luận", "giới thiệu", "phụ lục")
    _VI_NUMBER_WORD_BASE = "một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười một|mười hai"
    _VI_CHAPTER_NUMBER_PATTERN = rf"(?:{_ROMAN_OR_DIGIT_NUMBER}|{_VI_NUMBER_WORD_BASE})"
    _VI_RULES = _build_language_rules(
        numeric_aliases=_VI_ALIASES_NUMERIC,
        numeric_non_chapter_aliases=_VI_ALIASES_NUMERIC_NON_CHAPTER,
        chapter_aliases=("chương",),
        single_aliases=_VI_ALIASES_SINGLE,
        non_chapter_number_pattern=_ROMAN_OR_DIGIT_NUMBER,
        chapter_number_pattern=_VI_CHAPTER_NUMBER_PATTERN,
    )

    _ZH_NUMERIC_SECTION_ALIASES = ("章", "节", "節")
    _ZH_STRONG_SINGLE_ALIASES = ("序章", "终章", "終章", "附录", "附錄")
    _ZH_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:章)(?:[\s:：.\-–—].*)?\s*$"
            ),
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:节|節)(?:[\s:：.\-–—].*)?\s*$"
            ),
            re.compile(
                rf"^\s*卷\s*(?:{_CJK_NUMBER})(?:[\s:：.\-–—].*)?\s*$"
            ),
            _compile_single_heading_pattern(_ZH_STRONG_SINGLE_ALIASES),
        ),
        weak_heading_aliases=(
            "章",
            "节",
            "節",
            "卷",
            "序章",
            "终章",
            "終章",
            "附录",
            "附錄",
        ),
        weak_heading_signals=(
            "章",
            "节",
            "節",
            "卷",
            "序章",
            "终章",
            "終章",
            "附录",
            "附錄",
        ),
    )

    _JA_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:章|節)(?:[\s:：.\-–—].*)?\s*$"
            ),
            _compile_single_heading_pattern(("序章", "終章", "付録", "附録", "はじめに", "あとがき")),
        ),
        weak_heading_aliases=("章", "節", "序章", "終章", "付録", "附録", "はじめに", "あとがき"),
        weak_heading_signals=("章", "節", "序章", "終章", "付録", "附録", "はじめに", "あとがき"),
    )

    _KO_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(r"^\s*제\s*(?:\d+)\s*(?:장|절)(?:[\s:：.\-–—].*)?\s*$"),
            _compile_single_heading_pattern(("프롤로그", "에필로그", "서문", "부록")),
        ),
        weak_heading_aliases=("장", "절", "프롤로그", "에필로그", "서문", "부록"),
        weak_heading_signals=("장", "절", "프롤로그", "에필로그", "서문", "부록"),
    )

    _RU_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("глава", "раздел", "книга"), _ROMAN_OR_DIGIT_NUMBER),
            _compile_single_heading_pattern(("пролог", "эпилог", "введение", "приложение")),
        ),
        weak_heading_aliases=("глава", "раздел", "книга", "пролог", "эпилог", "введение", "приложение"),
        weak_heading_signals=("глава", "раздел", "книга", "пролог", "эпилог", "введение", "приложение"),
    )

    _UK_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("глава", "розділ", "книга"), _ROMAN_OR_DIGIT_NUMBER),
            _compile_single_heading_pattern(("пролог", "епілог", "вступ", "додаток")),
        ),
        weak_heading_aliases=("глава", "розділ", "книга", "пролог", "епілог", "вступ", "додаток"),
        weak_heading_signals=("глава", "розділ", "книга", "пролог", "епілог", "вступ", "додаток"),
    )

    _AR_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("الفصل", "القسم", "الكتاب"), _ARABIC_NUMBER),
            _compile_single_heading_pattern(("مقدمة", "خاتمة", "ملحق")),
        ),
        weak_heading_aliases=("الفصل", "القسم", "الكتاب", "مقدمة", "خاتمة", "ملحق"),
        weak_heading_signals=("الفصل", "القسم", "الكتاب", "مقدمة", "خاتمة", "ملحق"),
    )

    _HI_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("अध्याय", "भाग", "खंड"), _DEVANAGARI_NUMBER),
            _compile_single_heading_pattern(("प्रस्तावना", "उपसंहार", "परिशिष्ट")),
        ),
        weak_heading_aliases=("अध्याय", "भाग", "खंड", "प्रस्तावना", "उपसंहार", "परिशिष्ट"),
        weak_heading_signals=("अध्याय", "भाग", "खंड", "प्रस्तावना", "उपसंहार", "परिशिष्ट"),
    )

    _TH_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("บทที่", "ภาค", "ตอน"), r"\d+"),
            _compile_single_heading_pattern(("บทนำ", "บทส่งท้าย", "ภาคผนวก")),
        ),
        weak_heading_aliases=("บทที่", "ภาค", "ตอน", "บทนำ", "บทส่งท้าย", "ภาคผนวก"),
        weak_heading_signals=("บทที่", "ภาค", "ตอน", "บทนำ", "บทส่งท้าย", "ภาคผนวก"),
    )

    _TOC_MARKERS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = _ensure_all_language_entries(
        {
            LanguageCode.EN: (
                "contents",
                "table of contents",
                "contents page",
            ),
            LanguageCode.ZH: (
                "目录",
                "目錄",
                "章节目录",
                "章節目錄",
                "篇目",
                "总目录",
                "總目錄",
            ),
            LanguageCode.JA: (
                "目次",
                "目錄",
                "もくじ",
                "総目次",
            ),
            LanguageCode.KO: (
                "목차",
            ),
            LanguageCode.FR: (
                "sommaire",
                "table des matières",
            ),
            LanguageCode.DE: (
                "inhalt",
                "inhaltsverzeichnis",
            ),
            LanguageCode.ES: (
                "contenido",
                "índice",
                "tabla de contenido",
                "tabla de contenidos",
            ),
            LanguageCode.PT: (
                "conteúdo",
                "sumário",
                "índice",
            ),
            LanguageCode.IT: (
                "indice",
                "sommario",
            ),
            LanguageCode.RU: (
                "содержание",
                "оглавление",
            ),
            LanguageCode.AR: (
                "الفهرس",
                "جدول المحتويات",
                "المحتويات",
            ),
            LanguageCode.HI: (
                "विषय सूची",
                "अनुक्रमणिका",
            ),
            LanguageCode.TR: (
                "içindekiler",
            ),
            LanguageCode.NL: (
                "inhoudsopgave",
                "inhoud",
            ),
            LanguageCode.PL: (
                "spis treści",
            ),
            LanguageCode.UK: (
                "зміст",
            ),
            LanguageCode.ID: (
                "daftar isi",
            ),
            LanguageCode.VI: (
                "mục lục",
            ),
            LanguageCode.TH: (
                "สารบัญ",
            ),
        }
    )
    _FRONT_MATTER_MARKERS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = (
        _ensure_all_language_entries(
            {
                LanguageCode.EN: (
                    "preface",
                    "foreword",
                    "introduction",
                    "prologue",
                    "author's note",
                    "authors note",
                ),
                LanguageCode.ZH: (
                    "前言",
                    "序",
                    "序言",
                    "序文",
                    "导言",
                    "導言",
                    "引言",
                    "序章",
                ),
                LanguageCode.JA: (
                    "まえがき",
                    "序文",
                    "序",
                    "はじめに",
                    "プロローグ",
                    "序章",
                ),
                LanguageCode.KO: (
                    "머리말",
                    "서문",
                    "서론",
                    "프롤로그",
                ),
                LanguageCode.FR: (
                    "préface",
                    "avant-propos",
                    "introduction",
                    "prologue",
                ),
                LanguageCode.DE: (
                    "vorwort",
                    "einleitung",
                    "prolog",
                ),
                LanguageCode.ES: (
                    "prólogo",
                    "prefacio",
                    "introducción",
                ),
                LanguageCode.PT: (
                    "prefácio",
                    "prólogo",
                    "introdução",
                ),
                LanguageCode.IT: (
                    "prefazione",
                    "prologo",
                    "introduzione",
                ),
                LanguageCode.RU: (
                    "предисловие",
                    "введение",
                    "пролог",
                ),
                LanguageCode.AR: (
                    "مقدمة",
                    "تمهيد",
                    "مدخل",
                    "استهلال",
                ),
                LanguageCode.HI: (
                    "प्रस्तावना",
                    "भूमिका",
                    "परिचय",
                    "प्राक्कथन",
                ),
                LanguageCode.TR: (
                    "önsöz",
                    "giriş",
                    "prolog",
                ),
                LanguageCode.NL: (
                    "voorwoord",
                    "inleiding",
                    "proloog",
                ),
                LanguageCode.PL: (
                    "przedmowa",
                    "wstęp",
                    "prolog",
                ),
                LanguageCode.UK: (
                    "передмова",
                    "вступ",
                    "пролог",
                ),
                LanguageCode.ID: (
                    "kata pengantar",
                    "pendahuluan",
                    "prolog",
                ),
                LanguageCode.VI: (
                    "lời nói đầu",
                    "lời tựa",
                    "giới thiệu",
                    "mở đầu",
                ),
                LanguageCode.TH: (
                    "คำนำ",
                    "บทนำ",
                    "เกริ่นนำ",
                ),
            }
        )
    )
    _APPENDIX_MARKERS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = (
        _ensure_all_language_entries(
            {
                LanguageCode.EN: (
                    "appendix",
                    "appendices",
                    "appendix a",
                ),
                LanguageCode.ZH: (
                    "附录",
                    "附錄",
                ),
                LanguageCode.JA: (
                    "付録",
                    "附録",
                    "補遺",
                ),
                LanguageCode.KO: (
                    "부록",
                    "별첨",
                ),
                LanguageCode.FR: (
                    "annexe",
                    "annexes",
                ),
                LanguageCode.DE: (
                    "anhang",
                    "anhänge",
                ),
                LanguageCode.ES: (
                    "apéndice",
                    "anexo",
                    "anexos",
                ),
                LanguageCode.PT: (
                    "apêndice",
                    "anexo",
                    "anexos",
                ),
                LanguageCode.IT: (
                    "appendice",
                    "appendici",
                    "allegato",
                ),
                LanguageCode.RU: (
                    "приложение",
                    "приложения",
                ),
                LanguageCode.AR: (
                    "ملحق",
                    "ملاحق",
                ),
                LanguageCode.HI: (
                    "परिशिष्ट",
                    "परिशिष्टें",
                ),
                LanguageCode.TR: (
                    "ek",
                    "ekler",
                ),
                LanguageCode.NL: (
                    "bijlage",
                    "bijlagen",
                ),
                LanguageCode.PL: (
                    "aneks",
                    "załącznik",
                    "załączniki",
                ),
                LanguageCode.UK: (
                    "додаток",
                    "додатки",
                ),
                LanguageCode.ID: (
                    "lampiran",
                ),
                LanguageCode.VI: (
                    "phụ lục",
                ),
                LanguageCode.TH: (
                    "ภาคผนวก",
                ),
            }
        )
    )
    _BACK_MATTER_MARKERS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = (
        _ensure_all_language_entries(
            {
                LanguageCode.EN: (
                    "afterword",
                    "epilogue",
                    "postscript",
                ),
                LanguageCode.ZH: (
                    "后记",
                    "後記",
                    "跋",
                    "终章",
                    "終章",
                ),
                LanguageCode.JA: (
                    "あとがき",
                    "後書き",
                    "エピローグ",
                ),
                LanguageCode.KO: (
                    "후기",
                    "맺음말",
                    "에필로그",
                ),
                LanguageCode.FR: (
                    "postface",
                    "épilogue",
                ),
                LanguageCode.DE: (
                    "nachwort",
                    "epilog",
                ),
                LanguageCode.ES: (
                    "epílogo",
                    "posfacio",
                ),
                LanguageCode.PT: (
                    "epílogo",
                    "posfácio",
                ),
                LanguageCode.IT: (
                    "epilogo",
                    "postfazione",
                ),
                LanguageCode.RU: (
                    "послесловие",
                    "эпилог",
                ),
                LanguageCode.AR: (
                    "خاتمة",
                    "كلمة ختامية",
                ),
                LanguageCode.HI: (
                    "उपसंहार",
                    "पश्चलेख",
                ),
                LanguageCode.TR: (
                    "sonsöz",
                    "epilog",
                ),
                LanguageCode.NL: (
                    "nawoord",
                    "epiloog",
                ),
                LanguageCode.PL: (
                    "posłowie",
                    "epilog",
                ),
                LanguageCode.UK: (
                    "післямова",
                    "епілог",
                ),
                LanguageCode.ID: (
                    "epilog",
                    "penutup",
                ),
                LanguageCode.VI: (
                    "lời bạt",
                    "hậu ký",
                    "kết từ",
                ),
                LanguageCode.TH: (
                    "บทส่งท้าย",
                    "ปัจฉิมลิขิต",
                ),
            }
        )
    )
    _BODY_START_HEADING_HINTS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = {
        LanguageCode.EN: (
            "part",
            "chapter",
            "section",
            "book",
            "lesson",
            "unit",
        ),
        LanguageCode.ZH: (
            "第",
            "章",
            "节",
            "節",
            "卷",
            "序章",
            "终章",
            "終章",
        ),
        LanguageCode.JA: (
            "第",
            "章",
            "節",
            "部",
            "編",
            "巻",
            "卷",
        ),
        LanguageCode.KO: (
            "제",
            "장",
            "절",
            "부",
            "권",
        ),
        LanguageCode.FR: (
            "partie",
            "chapitre",
            "section",
            "livre",
        ),
        LanguageCode.DE: (
            "teil",
            "kapitel",
            "abschnitt",
            "buch",
        ),
        LanguageCode.ES: (
            "parte",
            "capítulo",
            "sección",
            "libro",
        ),
        LanguageCode.PT: (
            "parte",
            "capítulo",
            "seção",
            "livro",
        ),
        LanguageCode.IT: (
            "parte",
            "capitolo",
            "sezione",
            "libro",
        ),
        LanguageCode.RU: (
            "часть",
            "глава",
            "раздел",
            "книга",
        ),
        LanguageCode.AR: (
            "الجزء",
            "الفصل",
            "القسم",
            "الكتاب",
        ),
        LanguageCode.HI: (
            "भाग",
            "अध्याय",
            "खंड",
        ),
        LanguageCode.TR: (
            "bölüm",
            "kısım",
            "kitap",
            "ünite",
        ),
        LanguageCode.NL: (
            "deel",
            "hoofdstuk",
            "sectie",
            "boek",
        ),
        LanguageCode.PL: (
            "część",
            "rozdział",
            "sekcja",
            "księga",
        ),
        LanguageCode.UK: (
            "частина",
            "глава",
            "розділ",
            "книга",
        ),
        LanguageCode.ID: (
            "bagian",
            "bab",
        ),
        LanguageCode.VI: (
            "phần",
            "chương",
            "mục",
            "quyển",
        ),
        LanguageCode.TH: (
            "ภาค",
            "บทที่",
            "ตอน",
            "เล่ม",
        ),
    }
    _PART_HEADING_HINTS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = {
        LanguageCode.EN: ("part",),
        LanguageCode.ZH: ("卷",),
        LanguageCode.JA: ("部", "編", "巻", "卷"),
        LanguageCode.KO: ("부", "권"),
        LanguageCode.FR: ("partie",),
        LanguageCode.DE: ("teil",),
        LanguageCode.ES: ("parte",),
        LanguageCode.PT: ("parte",),
        LanguageCode.IT: ("parte",),
        LanguageCode.RU: ("часть",),
        LanguageCode.AR: ("الجزء",),
        LanguageCode.HI: ("भाग",),
        LanguageCode.TR: ("kısım", "bölüm"),
        LanguageCode.NL: ("deel",),
        LanguageCode.PL: ("część",),
        LanguageCode.UK: ("частина",),
        LanguageCode.ID: ("bagian",),
        LanguageCode.VI: ("phần",),
        LanguageCode.TH: ("ภาค",),
    }
    _CHAPTER_HEADING_HINTS_BY_LANGUAGE: dict[LanguageCode, tuple[str, ...]] = {
        LanguageCode.EN: ("chapter",),
        LanguageCode.ZH: ("第", "章", "节", "節"),
        LanguageCode.JA: ("第", "章", "節"),
        LanguageCode.KO: ("제", "장", "절"),
        LanguageCode.FR: ("chapitre",),
        LanguageCode.DE: ("kapitel",),
        LanguageCode.ES: ("capítulo",),
        LanguageCode.PT: ("capítulo",),
        LanguageCode.IT: ("capitolo",),
        LanguageCode.RU: ("глава",),
        LanguageCode.AR: ("الفصل",),
        LanguageCode.HI: ("अध्याय",),
        LanguageCode.TR: ("bölüm",),
        LanguageCode.NL: ("hoofdstuk",),
        LanguageCode.PL: ("rozdział",),
        LanguageCode.UK: ("глава",),
        LanguageCode.ID: ("bab",),
        LanguageCode.VI: ("chương",),
        LanguageCode.TH: ("บทที่",),
    }
    _EMPTY_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=tuple(),
        weak_heading_aliases=tuple(),
        weak_heading_signals=tuple(),
    )

    _RULES_BY_LANGUAGE: dict[LanguageCode, DocumentStructureLanguageRules] = {
        LanguageCode.EN: _EN_RULES,
        LanguageCode.ZH: _ZH_RULES,
        LanguageCode.JA: _JA_RULES,
        LanguageCode.KO: _KO_RULES,
        LanguageCode.FR: _FR_RULES,
        LanguageCode.DE: _DE_RULES,
        LanguageCode.ES: _ES_RULES,
        LanguageCode.PT: _PT_RULES,
        LanguageCode.IT: _IT_RULES,
        LanguageCode.RU: _RU_RULES,
        LanguageCode.AR: _AR_RULES,
        LanguageCode.HI: _HI_RULES,
        LanguageCode.TR: _TR_RULES,
        LanguageCode.NL: _NL_RULES,
        LanguageCode.PL: _PL_RULES,
        LanguageCode.UK: _UK_RULES,
        LanguageCode.ID: _ID_RULES,
        LanguageCode.VI: _VI_RULES,
        LanguageCode.TH: _TH_RULES,
    }

    @classmethod
    def is_language_supported(cls, language: LanguageCode) -> bool:
        """Return whether registry has explicit rules for this language."""
        return language in cls._RULES_BY_LANGUAGE

    @classmethod
    def get_rules(cls, language: LanguageCode) -> DocumentStructureLanguageRules:
        """Return language-specific rules, or safe empty fallback rules."""
        return cls._RULES_BY_LANGUAGE.get(language, cls._EMPTY_RULES)

    @classmethod
    def get_strong_heading_patterns(
        cls,
        language: LanguageCode,
    ) -> tuple[Pattern[str], ...]:
        """Return strong heading regex patterns for one language."""
        return cls.get_rules(language).strong_heading_patterns

    @classmethod
    def get_weak_heading_signals(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return weak heading aliases/signals for one language."""
        return cls.get_rules(language).weak_heading_signals

    @classmethod
    def get_weak_heading_aliases(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return weak heading aliases for one language."""
        return cls.get_rules(language).weak_heading_aliases

    @classmethod
    def get_toc_markers(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return TOC markers used for body-start detection."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._TOC_MARKERS_BY_LANGUAGE,
        )

    @classmethod
    def get_front_matter_markers(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return front-matter markers used for body-start detection."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._FRONT_MATTER_MARKERS_BY_LANGUAGE,
        )

    @classmethod
    def get_appendix_markers(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return appendix markers used for region-aware section-role classification."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._APPENDIX_MARKERS_BY_LANGUAGE,
        )

    @classmethod
    def get_back_matter_markers(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return back-matter markers used for region-aware section-role classification."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._BACK_MATTER_MARKERS_BY_LANGUAGE,
        )

    @classmethod
    def get_body_start_heading_hints(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return heading hints that help identify TOC heading lists and body restarts."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._BODY_START_HEADING_HINTS_BY_LANGUAGE,
        )

    @classmethod
    def get_part_heading_hints(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return language hints used to identify part-level headings."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._PART_HEADING_HINTS_BY_LANGUAGE,
        )

    @classmethod
    def get_chapter_heading_hints(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return language hints used to identify chapter-level headings."""
        return cls._resolve_text_markers(
            language=language,
            marker_map=cls._CHAPTER_HEADING_HINTS_BY_LANGUAGE,
        )

    @classmethod
    def _resolve_text_markers(
        cls,
        *,
        language: LanguageCode,
        marker_map: dict[LanguageCode, tuple[str, ...]],
    ) -> tuple[str, ...]:
        """Resolve language text-marker tuples with conservative fallback behavior."""
        return marker_map.get(language, tuple())
