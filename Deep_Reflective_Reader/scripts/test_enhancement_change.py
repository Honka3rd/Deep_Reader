"""Offline change classification; fixtures contain only synthetic source text."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_structure.enhancement_change import (
    EnhancementComparisonError, compare_enhancement_documents,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter, StructuredDocument, StructuredSection,
)


SOURCE = "First.\n\u7b2c\u4e8c\u6bb5\U0001f600\nCafe\u0301.\n"


def document(prefix="old", cuts=(0, 7, len(SOURCE))):
    sections = [
        StructuredSection(
            section_id=f"{prefix}-s{i}", section_index=i, title=f"Section {i}",
            level=2, content=SOURCE[start:end], char_start=start, char_end=end,
            parent_chapter_id=f"{prefix}-c", section_role=SectionRole.MAIN_BODY,
        )
        for i, (start, end) in enumerate(zip(cuts, cuts[1:]))
    ]
    return StructuredDocument(
        document_id="synthetic", title="Book", source_path=None, language="en",
        raw_text=SOURCE, chapters=[StructuredChapter(
            chapter_id=f"{prefix}-c", title="Chapter", level=1,
            chapter_role="main_body", sections=sections)],
    )


def with_section(doc, index=0, **changes):
    chapter = doc.chapters[0]
    sections = list(chapter.sections)
    sections[index] = replace(sections[index], **changes)
    return replace(doc, chapters=[replace(chapter, sections=sections)])


class EnhancementChangeTests(unittest.TestCase):
    def compare(self, candidate, existing=None):
        return compare_enhancement_documents(
            existing or document(), candidate, canonical_text=SOURCE)

    def test_ids_are_not_structure_and_mapping_preserves_old_identity(self):
        result = self.compare(document("new"))
        self.assertEqual(result.kind, "unchanged")
        self.assertEqual(
            [(n.existing_id, n.candidate_id) for n in result.correspondence],
            [("old-c", "new-c"), ("old-s0", "new-s0"), ("old-s1", "new-s1")])
        self.assertEqual(result.correspondence[2].section_index, 1)

    def test_section_title_only(self):
        result = self.compare(with_section(document("new"), title="New label"))
        self.assertEqual(result.kind, "titles_only")
        self.assertEqual(result.correspondence[1].existing_id, "old-s0")

    def test_chapter_title_only(self):
        doc = document()
        self.assertEqual(self.compare(replace(doc, chapters=[
            replace(doc.chapters[0], title="New chapter")])).kind, "titles_only")

    def test_label_representation_only_trims_edges(self):
        doc = with_section(document(), title=" Section 0 ")
        self.assertEqual(self.compare(doc).kind, "unchanged")
        self.assertEqual(self.compare(with_section(doc, title="Section  0")).kind, "titles_only")
        self.assertEqual(self.compare(with_section(doc, title="section 0")).kind, "titles_only")
        self.assertEqual(self.compare(
            with_section(doc, title=""), with_section(doc, title=None)).kind, "unchanged")

    def test_container_navigation_label(self):
        self.assertEqual(self.compare(with_section(
            document(), container_title="Part")).kind, "titles_only")

    def test_metadata_and_derived_units_do_not_change_structure(self):
        doc = document()
        candidate = replace(
            with_section(doc, section_index=99, section_kind="subsection",
                         is_implicit_section=True, task_units=[object()],
                         task_artifacts=object()),
            parse_provenance={"request_id": "new", "timestamp": "later"},
            source_path="new.pdf", title="Library title",
            document_task_artifacts=object(),
        )
        self.assertEqual(self.compare(candidate).kind, "unchanged")

    def test_postgres_without_raw_text_uses_exact_section_coverage(self):
        self.assertEqual(self.compare(document("new"),
                                     replace(document(), raw_text="")).kind, "unchanged")

    def test_boundary_change(self):
        result = self.compare(document(cuts=(0, 8, len(SOURCE))))
        self.assertEqual(result.kind, "structure_changed")
        self.assertEqual(result.correspondence, ())

    def test_split_and_merge(self):
        for cuts in [(0, len(SOURCE)), (0, 7, 10, len(SOURCE))]:
            with self.subTest(cuts=cuts):
                self.assertEqual(self.compare(document(cuts=cuts)).kind, "structure_changed")

    def test_level_changes(self):
        doc = document()
        for candidate in [with_section(doc, level=3),
                          replace(doc, chapters=[replace(doc.chapters[0], level=2)])]:
            self.assertEqual(self.compare(candidate).kind, "structure_changed")

    def test_parent_grouping_changes(self):
        doc = document()
        chapter = doc.chapters[0]
        candidate = replace(doc, chapters=[
            replace(chapter, sections=chapter.sections[:1]),
            replace(chapter, chapter_id="other", sections=[
                replace(chapter.sections[1], parent_chapter_id="other")]),
        ])
        self.assertEqual(self.compare(candidate).kind, "structure_changed")

    def test_target_roles_change_structure(self):
        doc = document()
        for candidate in [
            with_section(doc, section_role=SectionRole.TOC),
            replace(doc, chapters=[replace(doc.chapters[0], chapter_role="front_matter")]),
        ]:
            self.assertEqual(self.compare(candidate).kind, "structure_changed")

    def test_invalid_source_or_coverage_is_not_a_change_kind(self):
        doc = document()
        for candidate in [
            replace(doc, raw_text=SOURCE + "rewritten"),
            with_section(doc, content="rewritten"),
            with_section(doc, char_start=1),
            with_section(doc, char_end=8),
            with_section(doc, char_start=False),
            with_section(doc, index=1, char_start=6),
            replace(doc, chapters=[replace(doc.chapters[0], sections=doc.chapters[0].sections[:1])]),
            replace(doc, chapters=[replace(doc.chapters[0], sections=list(reversed(doc.chapters[0].sections)))]),
        ]:
            with self.subTest(candidate=candidate):
                with self.assertRaises(EnhancementComparisonError):
                    self.compare(candidate)

    def test_invalid_identity_parent_or_hierarchy_rejected(self):
        doc = document()
        for candidate in [
            replace(doc, document_id="other"),
            replace(doc, chapters=[]),
            replace(doc, chapters=doc.chapters * 2),
            with_section(doc, section_id="old-s1"),
            with_section(doc, parent_chapter_id="missing"),
            with_section(doc, level=0),
        ]:
            with self.assertRaises(EnhancementComparisonError):
                self.compare(candidate)

    def test_invalid_baseline_is_rejected_too(self):
        with self.assertRaises(EnhancementComparisonError):
            self.compare(document(), with_section(document(), content="wrong"))

    def test_unicode_is_exact_not_normalized(self):
        doc = document()
        section = doc.chapters[0].sections[1]
        with self.assertRaises(EnhancementComparisonError):
            self.compare(with_section(doc, index=1,
                                      content=section.content.replace("e\u0301", "\u00e9")))

    def test_diagnostics_are_counts_and_kind_not_book_data(self):
        self.assertEqual(self.compare(document()).diagnostics(), {
            "change_kind": "unchanged", "existing_chapter_count": 1,
            "existing_section_count": 2, "accepted_candidate_chapter_count": 1,
            "accepted_candidate_section_count": 2,
        })

    def test_no_input_mutation(self):
        old, new = document(), document("new")
        before = deepcopy((old, new))
        self.compare(new, old)
        self.assertEqual((old, new), before)

    def test_legacy_mirrors_are_not_a_source(self):
        doc = document()
        self.assertEqual(self.compare(replace(
            doc, sections=[object()], structure_nodes=[object()])).kind, "unchanged")
        with self.assertRaises(EnhancementComparisonError):
            self.compare(replace(doc, chapters=[], sections=doc.chapters[0].sections))

    def test_empty_source_and_errors_do_not_leak_content(self):
        with self.assertRaisesRegex(EnhancementComparisonError, "^missing_canonical_source$"):
            compare_enhancement_documents(document(), document(), canonical_text="")
        with self.assertRaisesRegex(EnhancementComparisonError, "^canonical_source_mismatch$"):
            self.compare(with_section(document(), content="private text"))


if __name__ == "__main__":
    unittest.main()
