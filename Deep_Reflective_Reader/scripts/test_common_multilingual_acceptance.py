"""COMMON-LANG-01 candidate/final parity, not just registry regex coverage."""
import unittest

from document_structure.heading_acceptance import heading_decision
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.text_normalization import normalize_ocr_text
from language.language_code import LanguageCode


HEADINGS = {
    "en": ("Chapter 1", "Chapter 2", "Preface"),
    "zh": ("第一章", "第二章", "前言"),
    "ja": ("第一章", "第二章", "はじめに"),
    "ko": ("제1장", "제2장", "서문"),
    "fr": ("Chapitre 1", "Chapitre 2", "Préface"),
    "de": ("Kapitel 1", "Kapitel 2", "Vorwort"),
    "es": ("Capítulo 1", "Capítulo 2", "Prólogo"),
    "pt": ("Capítulo 1", "Capítulo 2", "Prefácio"),
    "it": ("Capitolo 1", "Capitolo 2", "Prefazione"),
    "ru": ("Глава 1", "Глава 2", "Предисловие"),
    "uk": ("Глава 1", "Глава 2", "Передмова"),
    "ar": ("الفصل ١", "الفصل ٢", "مقدمة"),
    "hi": ("अध्याय १", "अध्याय २", "प्रस्तावना"),
    "tr": ("Bölüm 1", "Bölüm 2", "Önsöz"),
    "nl": ("Hoofdstuk 1", "Hoofdstuk 2", "Voorwoord"),
    "pl": ("Rozdział 1", "Rozdział 2", "Przedmowa"),
    "id": ("Bab 1", "Bab 2", "Pendahuluan"),
    "vi": ("Chương 1", "Chương 2", "Lời mở đầu"),
    "th": ("บทที่ 1", "บทที่ 2", "บทนำ"),
}
# Controlled body text isolates structural evidence from literary interpretation.
BODY = "The reader followed the river through the town and returned home."


class CommonMultilingualAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.splitter = CommonSectionSplitter()

    def assert_conserved(self, text, sections):
        self.assertEqual("".join(section.content for section in sections), text)
        for section in sections:
            self.assertEqual(section.content, text[section.char_start:section.char_end])

    def test_all_registered_languages_retain_candidates_and_hierarchy(self):
        self.assertEqual(set(HEADINGS), {code.value for code in LanguageCode if code != LanguageCode.UNKNOWN})
        for code, (first, second, _) in HEADINGS.items():
            for newline in ("\n", "\n\n", "\r\n"):
                with self.subTest(language=code, newline=repr(newline)):
                    language = LanguageCode(code)
                    text = newline.join((first, BODY, second, BODY))
                    candidates = self.splitter._detect_strong_headings(
                        self.splitter._build_line_infos(text), language)
                    self.assertEqual([c.title for c in candidates], [first, second])
                    sections = self.splitter.split(text, language, toc_hint="absent")
                    self.assertEqual([s.title for s in sections], [first, second])
                    self.assert_conserved(text, sections)
                    document = StructuredDocumentBuilder().build(document_id="fixture",
                        title="fixture", raw_text=text, language=language, toc_hint="absent")
                    self.assertIsNone(document.structure_error_code)
                    titles = {s.title for c in document.chapters for s in c.sections}
                    self.assertTrue({first, second}.issubset(titles))

    def test_reproduce_old_final_filter_disagreement(self):
        for language, title in ((LanguageCode.FR, "Chapitre 1"), (LanguageCode.DE, "Kapitel 1")):
            text = title + "\n" + BODY
            self.assertEqual(heading_decision(text, 0, title), "uncertain")
            self.assertEqual(heading_decision(text, 0, title,
                heading_kind=lambda t: self.splitter._heading_kind(t, language)), "accepted")

    def test_regions_and_false_headings(self):
        for code, (first, second, region) in HEADINGS.items():
            language = LanguageCode(code)
            with self.subTest(language=code, kind="region"):
                text = region + "\n" + BODY
                sections = self.splitter.split(text, language, toc_hint="absent")
                self.assertEqual(sections[0].title, normalize_ocr_text(region))
                self.assert_conserved(text, sections)
            for false in (first + " explains the author's intentions.",
                          '"' + first + '"', first + " .... 12",
                          "- " + first, "1. " + first):
                with self.subTest(language=code, false=false):
                    text = BODY + "\n" + false + "\n" + BODY
                    sections = self.splitter.split(text, language, toc_hint="absent")
                    self.assertFalse(any(s.title == false for s in sections))
                    self.assert_conserved(text, sections)

    def test_untitled_and_unknown_do_not_invent_structure(self):
        for code in HEADINGS:
            sections = self.splitter.split(BODY, LanguageCode(code))
            self.assertEqual(len(sections), 1)
            self.assertIsNone(sections[0].title)
        # Preserve the explicit unsupported-language contract, never assume English.
        with self.assertRaisesRegex(ValueError, "unsupported.*unknown"):
            self.splitter.split(BODY, LanguageCode.UNKNOWN)

    def test_region_prefix_in_prose_is_not_numbered_evidence(self):
        for code, title in (("en", "Introduction to symbolic logic"),
                            ("de", "Einleitung in die Logik"),
                            ("fr", "Introduction aux principes")):
            text = BODY + "\n" + title + "\n" + BODY
            sections = self.splitter.split(text, LanguageCode(code), toc_hint="absent")
            self.assertFalse(any(s.title == title for s in sections))
            self.assert_conserved(text, sections)

    def test_foreign_headings_can_still_use_independent_layout_evidence(self):
        text = "Chapitre 1\n\n" + BODY + "\n\nChapitre 2\n\n" + BODY
        sections = self.splitter.split(text, LanguageCode.EN, toc_hint="absent")
        self.assertEqual([s.title for s in sections], ["Chapitre 1", "Chapitre 2"])
        self.assert_conserved(text, sections)


if __name__ == "__main__":
    unittest.main()
