"""Prompt/application contracts, not a live-model semantic-quality benchmark."""
from pathlib import Path
import sys
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.section_splitter_selector import SectionSplitterSelector
from document_structure.structure_plan_contract import (
    EnhancementIntent, TocObservation, source_preview, PROMPT_VERSION,
)
from document_structure.common_parse_policy import TocHint
from language.language_code import LanguageCode
from test_structure_plan_application import Provider


TEXT = (
    "The council tested pumps beside the river. Engineers compared the water pressure.\n\n"
    "The repairs required funding. Residents debated a levy and approved the expenditure.\n\n"
    "Years later the village opened a school. Children learned to read in the new building.\n"
)


class AutonomyTests(unittest.TestCase):
    def test_generated_chapters_and_subsection_without_printed_headings(self):
        def plan(payload, context):
            return {**payload, "boundaries": [
                {"start_quote": TEXT[position:position + 24], "title": title, "origin": "generated",
                 "level": level, "role": "main_body"}
                for position, title, level in [(0, "Water supply", 1),
                    (TEXT.index("The repairs"), "Funding", 2),
                    (TEXT.index("Years later"), "Education", 1)]]}
        common = StructuredDocumentBuilder()
        before = common.build("fixture", "Fixture", TEXT, LanguageCode.EN, toc_hint="absent")
        self.assertEqual(len(before.chapters), 1)
        provider = Provider(plan)
        builder = StructuredDocumentBuilder(section_splitter_selector=SectionSplitterSelector(
            llm_splitter=LLMSectionSplitter(llm_provider=provider)))
        result = builder.build("fixture", "Fixture", TEXT, LanguageCode.EN,
                               parser_mode="llm_enhanced", toc_hint="absent")
        self.assertIsNone(result.structure_error_code)
        self.assertEqual([len(ch.sections) for ch in result.chapters], [2, 1])
        self.assertEqual("".join(s.content for ch in result.chapters for s in ch.sections), TEXT)
        self.assertEqual(result.parse_provenance["prompt_instruction_version"], PROMPT_VERSION)
        self.assertEqual([x["title_origin"] for x in result.parse_provenance["boundary_origins"]],
                         ["generated"] * 3)
        self.assertEqual(before, common.build("fixture", "Fixture", TEXT, LanguageCode.EN, toc_hint="absent"))
        self.assertEqual(provider.calls, 1)

    def test_coherent_single_body_is_valid_and_not_automatically_retried(self):
        text = "The seed needs water to germinate. Moist soil supplies that water."
        provider = Provider()
        sections, provenance = LLMSectionSplitter(llm_provider=provider).split_candidate_with_provenance(
            raw_text=text, language=LanguageCode.EN)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].content, text)
        self.assertEqual(provenance["source_coverage"], "complete")
        self.assertEqual(provider.calls, 1)

    def test_reference_cap_does_not_block_later_semantic_transition(self):
        text = " ".join(f"Observation {i:03d}." for i in range(100)) + " A new discussion begins."
        position = text.index("A new discussion")
        preview = source_preview(text, None)
        segment = next(s for s in preview["segments"] if s["source_start"] <= position < s["source_end"])
        self.assertNotIn(f"b{position}", [r["boundary_id"] for r in segment["boundary_references"]])
        def plan(payload, context):
            return {**payload, "boundaries": [
                {"start_quote": "Observation 000.", "title": "Observations", "origin": "generated", "level": 1, "role": "main_body"},
                {"start_quote": "A new discussion begins.",
                 "title": "Discussion", "origin": "generated", "level": 1, "role": "main_body"}]}
        sections, _ = LLMSectionSplitter(llm_provider=Provider(plan)).split_candidate_with_provenance(
            raw_text=text, language=LanguageCode.EN)
        self.assertEqual([s.char_start for s in sections], [0, position])
        self.assertEqual("".join(s.content for s in sections), text)

    def test_safe_accepted_plan_log_has_counts_not_source_or_titles(self):
        splitter = LLMSectionSplitter(llm_provider=Provider())
        with self.assertLogs("document_structure.llm_section_splitter", level="INFO") as logs:
            splitter.split_candidate_with_provenance(raw_text=TEXT, language=LanguageCode.EN)
        message = next(line for line in logs.output if "enhancement_plan_result" in line)
        self.assertIn("proposed_boundaries=1", message)
        self.assertIn("accepted_sections=1", message)
        self.assertIn(PROMPT_VERSION, message)
        self.assertNotIn("council", " ".join(logs.output))

    def test_full_source_navigation_and_output_allowance_audit(self):
        text = ("A distinct observation. Another complete sentence.\n" * 4000)[:147401]
        provider = Provider()
        preview, prompt, budget = LLMSectionSplitter(llm_provider=provider).assemble_candidate_request(
            text, LanguageCode.EN, EnhancementIntent.RESTORE_SOURCE, TocHint.ABSENT, TocObservation.NOT_CHECKED)
        from document_structure.semantic_structure_plan import read_semantic_prompt_context
        context = read_semantic_prompt_context(prompt)
        self.assertEqual(context["source"]["text"], text)
        self.assertEqual(len(preview["segments"]), 1)
        self.assertNotIn("boundary_references", context["source"])
        self.assertEqual(budget["output_reserve"], 4000)
        self.assertEqual(provider.calls, 0)
        # A representative 12-boundary JSON fits even a byte-upper-bound estimate.
        # This proves format feasibility, not that 12 divisions suit this source.
        plan = {"schema_version": "semantic_structure_plan_v1",
                "boundaries": [{"start_quote": f"Distinct topic opening {i}.", "title": f"Topic {i}",
                    "origin": "generated", "level": 1, "role": "main_body"} for i in range(12)]}
        self.assertLess(len(json.dumps(plan).encode()), 3276)


if __name__ == "__main__":
    unittest.main()
