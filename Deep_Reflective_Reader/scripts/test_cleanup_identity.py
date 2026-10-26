"""Reading overlays must not duplicate compatibility blocks after ID assignment."""
from dataclasses import replace
import unittest

from shared.task_unit_model import TaskUnit, TaskUnitContentBlock
from section_tasks.task_unit_id_normalizer import TaskUnitIdNormalizer
from section_tasks.task_unit_resolver import TaskUnitResolver
from scripts import test_enhancement_cleanup_api as api_fixture
from scripts.test_ocr_cleanup_validation import fixture


class UnitIdentityTests(unittest.TestCase):
    def unit(self):
        return TaskUnit("temporary", None, None, "First paragraph.\n\nSecond paragraph.", ["s0"], False)

    def test_generated_blocks_and_metadata_follow_unit_but_custom_ids_do_not(self):
        unit = self.unit()
        blocks = unit.segment_content_blocks() + [TaskUnitContentBlock("custom", "Independent note")]
        unit = replace(unit, content_blocks=blocks)
        rebound = unit.with_unit_id("42")
        self.assertEqual([b.block_id for b in rebound.content_blocks], ["42:content:0", "42:content:1", "custom"])
        self.assertEqual(rebound.content_blocks[0].metadata["content_block_id"], "42:content:0")
        self.assertEqual(unit.content_blocks[0].block_id, "temporary:content:0")
        self.assertEqual(rebound.content, unit.content)
        self.assertIs(rebound.with_unit_id("42"), rebound)

    def test_normalizer_rebinds_whole_content_compatibility_block(self):
        unit = self.unit()
        _, _, document = fixture(unit.content)
        section = document.chapters[0].sections[0]
        document = replace(document, chapters=[replace(document.chapters[0], sections=[
            replace(section, task_units=[unit])])])
        normalizer = TaskUnitIdNormalizer()
        changed = normalizer.normalize_document_task_unit_ids(document=document).chapters[0].sections[0].task_units[0]
        self.assertEqual(changed.content_blocks[0].block_id, changed.unit_id + ":content:0")
        mapped = normalizer.normalize_task_units_by_section_id(document=document,
            task_units_by_section_id={section.section_id: [unit]})[section.section_id][0]
        self.assertEqual(mapped.content_blocks[0].block_id, mapped.unit_id + ":content:0")


class PostgresCleanupIdentityTests(unittest.TestCase):
    setUpClass = classmethod(api_fixture.CleanupApiTests.setUpClass.__func__)
    drop_schema = classmethod(api_fixture.CleanupApiTests.drop_schema.__func__)

    def test_actual_api_layout_readback_no_duplicate_text_and_undo(self):
        from document_structure.reading_content import project_document, state_for
        from section_tasks.reading_target_resolver import ReadingTargetResolver
        case = api_fixture.CleanupApiTests()
        case.scoped = self.scoped
        self.addCleanup(case.doCleanups)
        case.setUp()
        case.kind = "structure_changed"
        case.coordinator.task_unit_resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=5000)
        case.assert_completed(case.execute(case.request()))
        document = case.f.snapshot()[0]
        for c in document.chapters:
            for s in c.sections:
                for u in s.task_units:
                    self.assertEqual(u.content_blocks[0].block_id, u.unit_id + ":content:0")
        actual = ReadingTargetResolver().resolve(document=document, target_level="document").content
        compact = lambda text: "".join(text.split())
        self.assertEqual(compact(actual), compact(case.f.text.replace("@@??", "")))
        projected = project_document(document)
        for c in projected.chapters:
            for s in c.sections:
                for u in s.task_units:
                    ids = [b.block_id for b in u.content_blocks]
                    self.assertEqual(len(ids), len(set(ids)))
        case.f.change("undo")
        restored = case.f.snapshot()[0]
        self.assertFalse(state_for(restored).get("cleanup_spans"))
        actual = ReadingTargetResolver().resolve(document=restored, target_level="document").content
        self.assertEqual(compact(actual), compact(case.f.text))


if __name__ == "__main__":
    unittest.main()
