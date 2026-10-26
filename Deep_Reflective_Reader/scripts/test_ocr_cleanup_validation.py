"""Offline location/safety tests; no semantic-accuracy or live-model claim."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest

from doc_loaders.extraction_identity import ExtractionResult
from document_structure.ocr_cleanup_contract import BoundCleanupProposal, CanonicalOcrSpan
from document_structure.ocr_cleanup_validation import (
    CleanupValidationError, validate_cleanup_proposal,
)
from document_structure.section_role import SectionRole
from document_structure.source_mapped_text import PageRange, SourceRange, build_source_mapped_text
from document_structure.structured_document import StructuredChapter, StructuredDocument, StructuredSection
from scripts.test_ocr_cleanup_contract import binding, evidence, item, proposal, source


BODY = "Meaningful reading content stays intact, including names and quotations.\n" * 15


def fixture(text=None, *, mode="ocr", cuts=None, origin="generated", representation="canonical_text_v1"):
    value = ExtractionResult(source(mode=mode).identity, text or "@@??\n" + BODY)
    cuts = cuts or (0, len(value.canonical_text))
    sections = [StructuredSection(
        section_id=f"s{index}", section_index=index, title="Heading", level=2,
        content=value.canonical_text[start:end], char_start=start, char_end=end,
        parent_chapter_id="c1",
    ) for index, (start, end) in enumerate(zip(cuts, cuts[1:]))]
    document = StructuredDocument(
        document_id="book", title="Book", source_path=None, language="zh",
        raw_text=value.canonical_text,
        chapters=[StructuredChapter(chapter_id="c1", title="Chapter", level=1,
                                    chapter_role="main_body", sections=sections)],
        parse_provenance={"boundary_origins": [
            {"char_start": section.char_start, "char_end": section.char_end, "title_origin": origin}
            for section in sections]},
    )
    return value, binding(value, source_representation=representation), document


class CleanupValidationTests(unittest.TestCase):
    def validate(self, items=None, *, fx=None, **kwargs):
        value, scope, document = fx or fixture()
        bound = BoundCleanupProposal.bind(source=value, binding=scope, proposal=proposal(
            [item(quote="@@??")] if items is None else items))
        return validate_cleanup_proposal(
            bound, source=value, expected_binding=scope, candidate=document, **kwargs)

    def reasons(self, result):
        return [entry.reason for entry in result.rejections]

    def test_exact_backend_spans_identity_and_no_mutation(self):
        fx = fixture()
        before = deepcopy(fx)
        result = self.validate(fx=fx)
        self.assertEqual(fx, before)
        deletion, = result.deletions
        self.assertEqual((deletion.char_start, deletion.char_end, deletion.chapter_id,
                          deletion.section_id, deletion.quote), (0, 4, "c1", "s0", "@@??"))
        self.assertEqual(result.proposed_count, 1)
        with self.assertRaises(FrozenInstanceError):
            deletion.char_end = 5
        self.assertNotIn("applied_count", result.diagnostics())

    def test_empty_plan(self):
        result = self.validate([])
        self.assertEqual((result.deletions, result.rejections, result.proposed_count), ((), (), 0))

    def test_repetition_requires_exact_adjacent_context(self):
        fx = fixture("First @@?? END\nSecond @@?? END\n" + BODY)
        self.assertEqual(self.reasons(self.validate(fx=fx)), ["ambiguous_quote"])
        result = self.validate([item(quote="@@??", prefix_quote="Second ", suffix_quote=" END")], fx=fx)
        self.assertEqual(result.deletions[0].char_start, 22)
        self.assertEqual(self.reasons(self.validate([
            item(quote="@@??", prefix_quote="Second\n")], fx=fx)), ["quote_not_found"])

    def test_context_occurrences_also_must_be_unique(self):
        fx = fixture("x@@??y x@@??y\n" + BODY)
        result = self.validate([item(quote="@@??", prefix_quote="x", suffix_quote="y")], fx=fx)
        self.assertEqual(self.reasons(result), ["ambiguous_quote"])

    def test_no_fuzzy_case_or_unicode_normalization(self):
        fx = fixture("Cafe\u0301\r\n@@??\n" + BODY)
        for quote in ("Caf\u00e9", "CAF", "@@? ?", "Cafe\u0301\n"):
            with self.subTest(quote=quote):
                self.assertEqual(self.reasons(self.validate([item(quote=quote)], fx=fx)), ["quote_not_found"])

    def test_unicode_code_points_and_crlf_preserved(self):
        quote = "\U0001f600\r\n@@??"
        result = self.validate([item(quote=quote)], fx=fixture(quote + "\n" + BODY))
        self.assertEqual(result.deletions[0].char_end, len(quote))
        self.assertEqual(result.deletions[0].quote, quote)

    def test_never_split_combining_token_or_crlf(self):
        for text, quote in (("abc\n", "bc"), ("e\u0301\n", "e"),
                            ("@@\r\n@@\n", "@@\r"),
                            ("\U0001f469\u200d\U0001f4bb\n", "\U0001f469")):
            with self.subTest(text=text):
                result = self.validate([item(quote=quote)], fx=fixture(text + "x" * 1000))
                self.assertEqual(self.reasons(result), ["unsafe_text_boundary"])

    def test_readable_identity_mapping(self):
        fx = fixture(representation="readable_full_source_v1")
        self.assertEqual(len(self.validate(fx=fx, readable_view=build_source_mapped_text(
            fx[0].canonical_text)).deletions), 1)

    def test_readable_offset_shift_and_unsafe_widening(self):
        text = "\u4e2d \u6587\n@@??\n" + BODY
        fx = fixture(text, representation="readable_full_source_v1")
        view = build_source_mapped_text(text, confirmed_fragment_ranges=(SourceRange(0, 3),))
        result = self.validate(fx=fx, readable_view=view)
        self.assertEqual(result.deletions[0].char_start, 4)
        result = self.validate([item(quote="\u4e2d\u6587")], fx=fx, readable_view=view)
        self.assertEqual(self.reasons(result), ["nonexact_canonical_mapping"])

    def test_forged_missing_or_stale_readable_mapping(self):
        fx = fixture(representation="readable_full_source_v1")
        view = build_source_mapped_text(fx[0].canonical_text)
        for invalid in (None, replace(view, source_offsets=tuple(reversed(view.source_offsets))),
                        replace(view, source_text_sha256="0" * 64)):
            with self.subTest(view=invalid is None), self.assertRaisesRegex(
                    CleanupValidationError, "^invalid_readable_mapping$"):
                self.validate(fx=fx, readable_view=invalid)

    def test_cross_page_evidence_is_clipped_and_not_a_locator(self):
        fx = fixture()
        pages = (PageRange(0, 0, 2), PageRange(1, 2, len(fx[0].canonical_text)))
        result = self.validate(fx=fx, page_ranges=pages)
        self.assertEqual(result.deletions[0].pages, (PageRange(0, 0, 2), PageRange(1, 2, 4)))
        repeated = fixture("@@??\n@@??\n" + BODY)
        self.assertEqual(self.reasons(self.validate(fx=repeated)), ["ambiguous_quote"])

    def test_invalid_page_maps_fail_closed(self):
        fx = fixture()
        length = fx[0].text_length
        for pages in ((PageRange(0, 1, length),), (PageRange(-1, 0, length),),
                      (PageRange(1, 0, 5), PageRange(1, 5, length)),
                      (PageRange(0, 0, length + 1),)):
            with self.subTest(pages=pages), self.assertRaisesRegex(CleanupValidationError, "invalid_page_mapping"):
                self.validate(fx=fx, page_ranges=pages)

    def test_cross_section_rejected(self):
        fx = fixture(cuts=(0, 2, len("@@??\n" + BODY)))
        self.assertEqual(self.reasons(self.validate(fx=fx)), ["cross_section"])

    def test_source_heading_not_generated_title_is_protected(self):
        self.assertEqual(self.reasons(self.validate(fx=fixture(origin="source"))), ["heading_anchor"])
        self.assertEqual(self.reasons(self.validate(fx=fixture("\n@@??\n" + BODY, origin="source"))),
                         ["heading_anchor"])
        self.assertEqual(len(self.validate(fx=fixture(origin="generated")).deletions), 1)

    def test_missing_or_invalid_boundary_evidence(self):
        value, scope, document = fixture()
        for origins in ([], [{"char_start": 0, "char_end": value.text_length, "title_origin": "guess"}]):
            with self.subTest(origins=origins), self.assertRaises(CleanupValidationError):
                self.validate(fx=(value, scope, replace(document, parse_provenance={"boundary_origins": origins})))

    def test_user_edits_win(self):
        self.assertEqual(self.reasons(self.validate(user_edit_ranges=(SourceRange(1, 3),))),
                         ["user_edited_content"])

    def test_meaningful_multilingual_retention_controls(self):
        for quote in ("Charles Bovary", "\u5b6b\u9686\u57fa", "\u53e4\u4f86\u5f81\u6230\u5e7e\u4eba\u56de", "E = mc^2", "1. 2026", "\u039a\u03b5\u03c6\u03ac\u03bb\u03b1\u03b9\u03bf"):
            with self.subTest(quote=quote):
                result = self.validate([item(quote=quote)], fx=fixture(quote + "\n" + BODY),
                                       protected_ranges=(SourceRange(0, len(quote)),))
                self.assertEqual(self.reasons(result), ["protected_content"])

    def test_duplicates_coalesce_without_mutating_count(self):
        result = self.validate([item(quote="@@??"), item(quote="@@??", reason="recognition_fragment")])
        self.assertEqual(len(result.deletions), 1)
        self.assertEqual(self.reasons(result), ["duplicate_deletion"])
        self.assertEqual(result.proposed_count, len(result.deletions) + len(result.rejections))

    def test_nonidentical_overlaps_reject_all_including_duplicates(self):
        for quotes in (("@@?", "@??", "@@?"), ("@??", "@@?")):
            result = self.validate([item(quote=quote) for quote in quotes])
            self.assertFalse(result.deletions)
            self.assertEqual(self.reasons(result), ["overlapping_deletion"] * len(quotes))

    def test_adjacent_deletions_are_not_overlaps(self):
        result = self.validate([item(quote="@@"), item(quote="??")])
        self.assertEqual(len(result.deletions), 2)

    def test_bad_item_does_not_discard_safe_subset(self):
        result = self.validate([item(quote="@@??"), item(quote="missing")])
        self.assertEqual(len(result.deletions), 1)
        self.assertEqual(self.reasons(result), ["quote_not_found"])

    def test_all_noise_document_is_not_erased(self):
        result = self.validate(fx=fixture("@@??"))
        self.assertEqual(self.reasons(result), ["all_content_removal"])
        self.assertFalse(result.deletions)

    def test_front_matter_padding_cannot_allow_erasing_entire_body(self):
        value, scope, document = fixture(cuts=(0, 5, len("@@??\n" + BODY)))
        chapter = document.chapters[0]
        sections = [replace(chapter.sections[0], section_role=SectionRole.MAIN_BODY),
                    replace(chapter.sections[1], section_role=SectionRole.FRONT_MATTER)]
        document = replace(document, chapters=[replace(chapter, sections=sections)])
        result = self.validate(fx=(value, scope, document))
        self.assertEqual(self.reasons(result), ["all_body_removal"])

    def test_invalid_protected_range_fails_whole_plan(self):
        for span in (SourceRange(-1, 2), SourceRange(2, 2), SourceRange(0, 100000)):
            with self.subTest(span=span), self.assertRaisesRegex(CleanupValidationError, "invalid_protected_ranges"):
                self.validate(protected_ranges=(span,))

    def test_no_percent_limit_including_nonspace_ratio(self):
        for text in ("@@??\n" + "x" * 36, "@@??\n" + "x" * 35, "@@??" + " " * 1000 + "x"):
            result = self.validate(fx=fixture(text))
            self.assertEqual(len(result.deletions), 1)
            self.assertFalse(result.rejections)

    def test_no_absolute_limit_or_order_dependent_truncation(self):
        chunks = [f"@{index:02d}" + "?" * 509 for index in range(9)]
        for count in (8, 9):
            result = self.validate([item(quote=quote) for quote in chunks[:count]],
                fx=fixture("\n".join(chunks[:count]) + "\n" + "x" * 50000))
            self.assertEqual(sum(len(entry.quote) for entry in result.deletions), count * 512)
            self.assertFalse(result.rejections)

    def test_large_batch_and_long_quote_keep_valid_subset(self):
        chunks = [f"@{index:03d}" + "?" * 513 for index in range(129)]
        result = self.validate([item(quote=q) for q in chunks] + [item(quote="missing")],
            fx=fixture("\n".join(chunks) + "\nRetained body."))
        self.assertEqual([entry.quote for entry in result.deletions], chunks)
        self.assertEqual(self.reasons(result), ["quote_not_found"])

    def test_source_and_all_scope_preconditions_rechecked(self):
        value, scope, document = fixture()
        bound = BoundCleanupProposal.bind(source=value, binding=scope, proposal=proposal([item(quote="@@??")]))
        changes = {"namespace": "other", "document_id": "other", "operation_id": "other",
                   "structure_version": 2, "structure_fingerprint": "c" * 64,
                   "expected_reading_revision": 1, "prompt_version": "new"}
        for field, replacement in changes.items():
            with self.subTest(field=field), self.assertRaisesRegex(CleanupValidationError, "cleanup_binding_mismatch"):
                validate_cleanup_proposal(bound, source=value, candidate=document,
                    expected_binding=scope.model_copy(update={field: replacement}))
        with self.assertRaisesRegex(CleanupValidationError, "cleanup_source_mismatch"):
            validate_cleanup_proposal(bound, source=replace(value, canonical_text=value.canonical_text + "!"),
                                      candidate=document, expected_binding=scope)

    def test_invalid_candidate_identity_or_source(self):
        value, scope, document = fixture()
        for changed in (replace(document, document_id="other"), replace(document, raw_text="other"),
                        replace(document, chapters=[])):
            with self.subTest(changed=changed.document_id), self.assertRaises(CleanupValidationError):
                self.validate(fx=(value, scope, changed))

    def test_native_not_authorized_and_auto_must_have_actual_evidence(self):
        for mode, code in (("fast", "native_source"), ("auto", "ocr_evidence_missing")):
            with self.subTest(mode=mode), self.assertRaisesRegex(CleanupValidationError, code):
                self.validate(fx=fixture(mode=mode))

    def test_auto_scope_and_adjacent_ocr_pages(self):
        fx = fixture(mode="auto")
        spans = (CanonicalOcrSpan(char_start=0, char_end=2, page_index=0),
                 CanonicalOcrSpan(char_start=2, char_end=4, page_index=1))
        self.assertEqual(len(self.validate(fx=fx, evidence=evidence(fx[0], spans)).deletions), 1)
        for incomplete in (spans[:1], (spans[0], CanonicalOcrSpan(char_start=3, char_end=4))):
            self.assertEqual(self.reasons(self.validate(fx=fx, evidence=evidence(fx[0], incomplete))),
                             ["outside_ocr_scope"])

    def test_diagnostics_do_not_leak_text_or_ids(self):
        result = self.validate([item(quote="@@??"), item(quote="PRIVATE-QUOTE")])
        diagnostic = repr(result.diagnostics())
        for value in ("@@??", "PRIVATE-QUOTE", "book", "op-1"):
            self.assertNotIn(value, diagnostic)


if __name__ == "__main__":
    unittest.main()
