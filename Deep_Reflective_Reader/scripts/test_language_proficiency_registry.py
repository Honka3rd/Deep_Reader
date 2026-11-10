"""LLRN-01: curated reading levels, strict selections and registry evolution."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import unittest
from unittest.mock import patch

from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from language import language_proficiency_registry as definitions


Registry = definitions.LanguageProficiencyRegistry
SelectionError = definitions.ProficiencySelectionError
ConfigurationError = definitions.ProficiencyConfigurationError


class LanguageProficiencyRegistryTests(unittest.TestCase):
    def resolve(self, code="en", level="B1", **changes):
        scheme = Registry.get_scheme(code)
        values = dict(language_code=code, scheme_id=scheme.scheme_id,
                      scheme_version=scheme.version, level_id=level)
        return Registry.resolve_selection(**(values | changes))

    def test_every_registered_language_has_one_complete_definition(self):
        expected = tuple(c for c in LanguageProfileRegistry.get_supported_language_codes()
                         if c != "unknown")
        items = Registry.list_definitions()
        self.assertEqual(tuple(item.language_code for item in items), expected)
        self.assertEqual(len(set(expected)), len(items))
        for item in items:
            with self.subTest(language=item.language_code):
                self.assertTrue(item.scheme.scope_note)
                self.assertTrue(item.scheme.label)
                self.assertEqual([level.order for level in item.scheme.levels],
                                 list(range(1, len(item.scheme.levels) + 1)))
                for level in item.scheme.levels:
                    resolved = self.resolve(item.language_code, level.level_id)
                    self.assertEqual(resolved.level, level)
                    self.assertTrue(level.label.strip())
                    self.assertTrue(level.reading_description.strip())

    def test_language_specific_level_orders(self):
        expected = {
            "en": ("cefr_reading", ("A1", "A2", "B1", "B2", "C1", "C2")),
            "ja": ("jlpt_reading", ("N5", "N4", "N3", "N2", "N1")),
            "ko": ("topik_reading", ("1", "2", "3", "4", "5", "6")),
            "zh": ("tocfl_reading", ("1", "2", "3", "4", "5", "6")),
        }
        for code, (scheme_id, levels) in expected.items():
            with self.subTest(code=code):
                scheme = Registry.get_scheme(code)
                self.assertEqual(scheme.scheme_id, scheme_id)
                self.assertEqual(tuple(level.level_id for level in scheme.levels), levels)
        self.assertLess(self.resolve("ja", "N5").level.order,
                        self.resolve("ja", "N1").level.order)

    def test_product_scale_is_explicit_and_not_a_fabricated_exam(self):
        for code in ("ar", "hi", "id", "vi", "th"):
            scheme = Registry.get_scheme(code)
            self.assertEqual(scheme.scheme_id, "deep_reader_reading")
            self.assertIn("Product-defined", scheme.scope_note)
            self.assertEqual(scheme.reference_urls, ())
            self.assertEqual(tuple(level.level_id for level in scheme.levels),
                             ("beginner", "elementary", "intermediate", "advanced", "proficient"))
            with self.assertRaises(SelectionError):
                self.resolve(code, "A1")

    def test_exact_codes_only_no_unknown_alias_or_region_fallback(self):
        for code in ("unknown", "xx", "EN", " en", "en ", "english", "en-US",
                     "zh-Hant", "", None, 1, True, [], {}):
            with self.subTest(code=code), self.assertRaises(SelectionError) as raised:
                Registry.get_scheme(code)
            self.assertEqual(raised.exception.code, "proficiency_language_invalid")
        self.assertEqual(Registry.get_scheme(LanguageCode.EN), Registry.get_scheme("en"))

    def test_cross_scheme_selection_is_rejected_even_for_shared_number(self):
        with self.assertRaises(SelectionError) as raised:
            self.resolve("zh", "1", scheme_id="topik_reading")
        self.assertEqual(raised.exception.code, "proficiency_scheme_mismatch")
        with self.assertRaises(SelectionError):
            self.resolve("ja", "A1", scheme_id="cefr_reading")

    def test_level_ids_are_not_coerced_or_inferred(self):
        for level in (1, True, None, [], {}, "n5", " N5", "N6", "A1", ""):
            with self.subTest(level=level), self.assertRaises(SelectionError) as raised:
                self.resolve("ja", level)
            self.assertEqual(raised.exception.code, "proficiency_level_invalid")
        with self.assertRaises(SelectionError):
            self.resolve("zh", 1)

    def test_scheme_and_version_types_are_strict(self):
        for field, expected in (("scheme_id", "proficiency_scheme_mismatch"),
                                ("scheme_version", "proficiency_version_mismatch")):
            for value in (None, [], {}, 1, True, "", " 1", "999"):
                with self.subTest(field=field, value=value), self.assertRaises(SelectionError) as raised:
                    self.resolve(**{field: value})
                self.assertEqual(raised.exception.code, expected)

    def test_new_definition_version_rejects_old_selection(self):
        old = self.resolve()
        upgraded = replace(old.scheme, version="2", levels=(
            replace(old.scheme.levels[0], reading_description="Revised reading guidance."),
            *old.scheme.levels[1:],
        ))
        with patch.object(definitions, "_SCHEMES", definitions._SCHEMES | {upgraded.scheme_id: upgraded}):
            Registry.validate_coverage()
            with self.assertRaises(SelectionError) as raised:
                self.resolve(scheme_version="1")
            self.assertEqual(raised.exception.code, "proficiency_version_mismatch")
            new = self.resolve(scheme_version="2")
            self.assertNotEqual(old.identity, new.identity)
            self.assertEqual(new.identity, ("en", "cefr_reading", "2", "B1"))
        self.assertEqual(self.resolve().identity, old.identity)

    def test_identity_does_not_alias_languages_levels_or_schemes(self):
        selections = [self.resolve("en", "B1"), self.resolve("fr", "B1"),
                      self.resolve("en", "B2"), self.resolve("zh", "1"),
                      self.resolve("ko", "1")]
        self.assertEqual(len({value.identity for value in selections}), len(selections))

    def test_definitions_and_results_are_immutable(self):
        resolved = self.resolve()
        for obj, attr, value in ((resolved, "language_code", "ja"),
                                 (resolved.scheme, "version", "9"),
                                 (resolved.level, "label", "changed")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, attr, value)
        with self.assertRaises(TypeError):
            definitions._LANGUAGE_SCHEMES["en"] = "jlpt_reading"
        with self.assertRaises(TypeError):
            definitions._SCHEMES["cefr_reading"] = None
        projection = asdict(resolved)
        projection["scheme"]["version"] = "changed"
        self.assertEqual(Registry.get_scheme("en").version, "1")

    def test_data_can_be_serialized_without_provider_or_api_types(self):
        payload = json.loads(json.dumps([asdict(d) for d in Registry.list_definitions()], ensure_ascii=False))
        japanese = next(item for item in payload if item["language_code"] == "ja")
        self.assertEqual(japanese["scheme"]["levels"][0]["level_id"], "N5")
        self.assertIn("reading_description", japanese["scheme"]["levels"][0])

    def test_registry_expansion_requires_curated_mapping_no_silent_default(self):
        codes = LanguageProfileRegistry.get_supported_language_codes()
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=(*codes, "xx")):
            with self.assertRaises(ConfigurationError):
                Registry.list_definitions()
            with self.assertRaises(ConfigurationError):
                Registry.get_scheme("xx")

    def test_removed_language_cannot_remain_selectable_via_assignment(self):
        codes = tuple(c for c in LanguageProfileRegistry.get_supported_language_codes() if c != "ja")
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=codes):
            with self.assertRaises(SelectionError):
                Registry.get_scheme("ja")
            with self.assertRaises(ConfigurationError):
                Registry.validate_coverage()

    def test_registry_unknown_is_never_a_proficiency_option(self):
        codes = LanguageProfileRegistry.get_supported_language_codes()
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=(*codes, "unknown")):
            self.assertNotIn("unknown", tuple(d.language_code for d in Registry.list_definitions()))

    def test_duplicate_registry_entry_is_configuration_error(self):
        codes = LanguageProfileRegistry.get_supported_language_codes()
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=(*codes, "en")):
            with self.assertRaises(ConfigurationError):
                Registry.list_definitions()

    def test_missing_scheme_is_configuration_error(self):
        with patch.object(definitions, "_SCHEMES", {}):
            with self.assertRaises(ConfigurationError):
                Registry.list_definitions()
            with self.assertRaises(ConfigurationError):
                Registry.get_scheme("en")

    def test_invalid_curated_definition_cannot_be_listed(self):
        scheme = Registry.get_scheme("en")
        variants = (
            replace(scheme, version=""),
            replace(scheme, scheme_id="different"),
            replace(scheme, levels=()),
            replace(scheme, levels=(scheme.levels[0], scheme.levels[0])),
            replace(scheme, levels=tuple(reversed(scheme.levels))),
            replace(scheme, levels=(replace(scheme.levels[0], reading_description=" "),)),
        )
        for invalid in variants:
            with self.subTest(invalid=invalid), patch.object(
                    definitions, "_SCHEMES", definitions._SCHEMES | {scheme.scheme_id: invalid}):
                with self.assertRaises(ConfigurationError):
                    Registry.list_definitions()

    def test_invalid_user_values_are_not_echoed(self):
        private_value = "private caller text"
        for field in ("language_code", "scheme_id", "scheme_version", "level_id"):
            with self.subTest(field=field), self.assertRaises(SelectionError) as raised:
                self.resolve(**{field: private_value})
            self.assertNotIn(private_value, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
