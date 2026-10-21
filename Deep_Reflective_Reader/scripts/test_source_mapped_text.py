import hashlib
import unittest
from unittest.mock import patch

from document_structure.source_mapped_text import (
    PageRange, SourceRange, build_source_mapped_text,
)


ZH = "\u6751\u6c11\u8ba8\u8bba\u79e9\u5e8f"


def confirmed(text, **kwargs):
    return build_source_mapped_text(
        text, confirmed_fragment_ranges=(SourceRange(0, len(text)),), **kwargs)


class SourceMappedTextTests(unittest.TestCase):
    def test_default_preserves_ambiguous_heading_poetry_and_list(self):
        for text in ("\u5e8f\n" + ZH, "\n".join(ZH), "1. A\n2. B", "", " \r\n"):
            with self.subTest(text=text):
                view = build_source_mapped_text(text)
                self.assertEqual(view.text, text)
                self.assertEqual(view.source_offsets, tuple(range(len(text))))
                self.assertFalse(view.removed_ranges)

    def test_confirmed_cjk_fragmentation_maps_back_exactly(self):
        raw = "\n".join(ZH)
        view = confirmed(raw)
        self.assertEqual(view.text, ZH)
        self.assertEqual(view.source, raw)
        self.assertEqual(view.source_text_sha256, hashlib.sha256(raw.encode()).hexdigest())
        for start in range(len(view.text)):
            for end in range(start + 1, len(view.text) + 1):
                span = view.source_range(start, end)
                self.assertEqual(raw[span.start:span.end].replace("\n", ""), view.text[start:end])
        self.assertEqual("".join(raw[s.start:s.end] for s in view.line_ranges), raw)

    def test_crlf_is_one_line_break_but_blank_line_is_barrier(self):
        self.assertEqual(confirmed(ZH[0] + "\r\n" + ZH[1]).text, ZH[:2])
        for separator in ("\n\n", "\r\n \r\n", "\f", "\u2029", "\t", "\u2028"):
            raw = ZH[0] + separator + ZH[1]
            self.assertEqual(confirmed(raw).text, raw)

    def test_latin_numbers_punctuation_and_unicode_stay_unchanged(self):
        for raw in ("alpha beta", "A\nB", "12 34", ZH[0] + "\nA",
                    ZH[0] + "\n\u3002\n" + ZH[1], "\ufb01 \uff21"):
            self.assertEqual(confirmed(raw).text, raw)

    def test_only_confirmed_region_is_changed(self):
        raw = "\u5e8f\n\n" + "\n".join(ZH)
        view = build_source_mapped_text(raw, confirmed_fragment_ranges=(SourceRange(3, len(raw)),))
        self.assertEqual(view.text, "\u5e8f\n\n" + ZH)

    def test_page_evidence_and_separator_preserved(self):
        raw = ZH[0] + "\n" + ZH[1] + "\n" + ZH[2] + "\n" + ZH[3]
        pages = (PageRange(0, 0, 3), PageRange(1, 4, 7))
        view = build_source_mapped_text(raw, page_ranges=pages,
                                       confirmed_fragment_ranges=(SourceRange(0, 3), SourceRange(4, 7)))
        self.assertEqual(view.text, ZH[:2] + "\n" + ZH[2:4])
        self.assertEqual(view.page_ranges, pages)
        self.assertEqual(view.source_page_ranges(0, len(view.text)), pages)
        with self.assertRaises(ValueError):
            confirmed(raw, page_ranges=pages)

    def test_invalid_or_incomplete_page_evidence_rejected(self):
        for pages in ((PageRange(0, 1, 3),), (PageRange(0, 0, 1),),
                      (PageRange(1, 0, 1), PageRange(0, 1, 3)),
                      (PageRange(0, 0, 2), PageRange(1, 1, 3))):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                build_source_mapped_text("abc", page_ranges=pages)

    def test_empty_page_retains_identity(self):
        pages = (PageRange(0, 0, 0), PageRange(1, 0, 2))
        view = confirmed(ZH[:2], page_ranges=pages)
        self.assertEqual(view.page_ranges, pages)

    def test_invalid_fragment_and_analysis_intervals_rejected(self):
        for ranges in ((SourceRange(-1, 2),), (SourceRange(0, 4),),
                       (SourceRange(0, 0),), (SourceRange(0, 2), SourceRange(1, 3))):
            with self.assertRaises(ValueError):
                build_source_mapped_text("abc", confirmed_fragment_ranges=ranges)
        view = build_source_mapped_text("abc")
        for start, end in ((0, 0), (-1, 1), (0, 4), (2, 1), (0.5, 1), (False, 1)):
            with self.assertRaises(ValueError):
                view.source_range(start, end)

    def test_limits_fail_explicitly_without_truncation(self):
        with patch("document_structure.source_mapped_text.MAX_SOURCE_CHARS", 2):
            with self.assertRaises(ValueError):
                build_source_mapped_text("abc")
        with patch("document_structure.source_mapped_text.MAX_EVIDENCE_RANGES", 0):
            with self.assertRaises(ValueError):
                confirmed(ZH)

    def test_no_subprocess_or_network_dependency(self):
        with patch("subprocess.run", side_effect=AssertionError("subprocess")), \
                patch("socket.socket", side_effect=AssertionError("network")):
            self.assertEqual(confirmed("\n".join(ZH)).text, ZH)


if __name__ == "__main__":
    unittest.main()
