"""Registry coverage, capability failures and recognition-hint isolation."""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from doc_loaders import pdf_ocr_language_policy as policy
from language.language_profile_registry import LanguageProfileRegistry
from scripts import tokenizer_startup, verify_ocr_assets as assets


class OcrLanguageRegistryTests(unittest.TestCase):
    def test_all_registered_languages_have_correct_engine_mapping(self):
        expected = dict(en="eng", zh="chi_sim+chi_tra+eng", ja="jpn", ko="kor",
                        es="spa", pt="por", fr="fra", de="deu", it="ita", ru="rus",
                        ar="ara", hi="hin", tr="tur", nl="nld", pl="pol", uk="ukr",
                        id="ind", vi="vie", th="tha")
        self.assertEqual(set(expected), set(LanguageProfileRegistry.get_supported_language_codes()))
        for code, model in expected.items():
            with self.subTest(code=code):
                self.assertEqual(policy.get_tesseract_language_for_document_language(code), model)

    def test_default_and_unknown_follow_registry_order(self):
        for value in (None, "", "  ", "registry", " REGISTRY "):
            self.assertEqual(policy.normalize_tesseract_language_config(value),
                             "+".join(policy.registered_ocr_models()))
        self.assertEqual(policy.get_tesseract_language_for_document_language(None),
                         policy.DEFAULT_TESSERACT_OCR_LANGUAGE)
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=("fr", "de")):
            self.assertEqual(policy.normalize_tesseract_language_config(None), "fra+deu")
            self.assertEqual(policy.required_ocr_models(), ("fra", "deu", "osd"))

    def test_unmapped_new_registry_language_is_not_silently_english(self):
        with patch.object(LanguageProfileRegistry, "get_supported_language_codes", return_value=("en", "xx")):
            for operation in (policy.registered_ocr_models, policy.required_ocr_models,
                              lambda: policy.normalize_tesseract_language_config("en")):
                with self.assertRaisesRegex(ValueError, "ocr_registry_mapping_missing:xx"):
                    operation()

    def test_explicit_codes_legacy_models_and_ordered_deduplication(self):
        for value, expected in (("fr+de", "fra+deu"), ("zh+en", "chi_sim+chi_tra+eng"),
                                (" eng + fr +fra ", "eng+fra"), ("chi_tra_vert+eng", "chi_tra_vert+eng")):
            self.assertEqual(policy.normalize_tesseract_language_config(value), expected)

    def test_invalid_config_rejected_without_echoing_input(self):
        for value in ("secret-value", "fr+", "+en", "en++fr", "registry+fr", "osd", "../eng", "fra,eng"):
            with self.assertRaisesRegex(ValueError, "ocr_language_config_invalid") as error:
                policy.normalize_tesseract_language_config(value)
            self.assertNotIn(value, str(error.exception))

    def test_narrow_hint_does_not_change_installed_capabilities(self):
        with patch.dict(os.environ, {"DEEP_READER_PDF_OCR_LANGUAGE": "en"}):
            required = policy.required_ocr_models()
            self.assertIn("fra", required)
            self.assertIn("jpn", required)
            self.assertIn("chi_sim_vert", required)
            self.assertEqual(len(required), len(set(required)))

    def test_loader_expands_env_and_explicit_override(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        with patch.dict(os.environ, {"DEEP_READER_PDF_OCR_LANGUAGE": "fr+de"}):
            self.assertEqual(PdfDocumentLoader().ocr_language, "fra+deu")
            self.assertEqual(PdfDocumentLoader(ocr_language="ja").ocr_language, "jpn")

    def test_only_ocr_identity_changes_with_languages(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        from doc_loaders.pdf_extraction_identity import identity_for_loader
        a, b = PdfDocumentLoader(ocr_language="eng"), PdfDocumentLoader(ocr_language="registry")
        with patch("doc_loaders.pdf_extraction_identity.shutil.which", return_value=None):
            for mode in ("fast", "ocr", "auto"):
                first = identity_for_loader(a, source_sha256="a" * 64, mode=mode)
                second = identity_for_loader(b, source_sha256="a" * 64, mode=mode)
                self.assertEqual(first == second, mode == "fast")

    def test_registered_scripts_are_text_not_symbol_garbage(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        samples = (
            "科学的な研究には証拠と慎重な検討が必要です。",
            "과학 연구에는 증거와 신중한 검토가 필요합니다.",
            "Научные исследования требуют доказательств и тщательного анализа.",
            "Наукові дослідження потребують доказів і ретельного аналізу.",
            "البحث العلمي يحتاج إلى أدلة وتحليل دقيق ومراجعة مستقلة.",
            "वैज्ञानिक अनुसंधान के लिए प्रमाण और सावधानीपूर्वक अध्ययन आवश्यक है।",
            "การวิจัยทางวิทยาศาสตร์ต้องอาศัยหลักฐานและการวิเคราะห์อย่างรอบคอบ",
            "Les élèves étudient les phénomènes avec précision et curiosité.",
            "Nghiên cứu khoa học cần bằng chứng và phân tích cẩn thận.",
        )
        loader = PdfDocumentLoader(ocr_language="registry")
        for sample in samples:
            with self.subTest(sample=sample):
                candidate = loader._score_ocr_candidate(psm="6", text=sample * 3)
                self.assertTrue(candidate.passed, candidate.reasons)
                self.assertLess(candidate.symbol_ratio, 0.1)
        self.assertFalse(loader._score_ocr_candidate(psm="6", text="\u0301" * 100).passed)

    def test_adjacent_han_does_not_hide_complete_latin_words(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        loader = PdfDocumentLoader(ocr_language="registry")
        mixed = ("甲Scientific乙research丙requires丁evidence戊and己careful庚analysis " * 4
                 + " A B C D E F G H I J ")
        score = loader._score_ocr_candidate(psm="6", text=mixed)
        self.assertNotIn("fragmented_latin", score.reasons)

    def test_chinese_refinement_requires_strong_observed_consensus(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        from doc_loaders.chinese_ocr_strategy import horizontal_refinement_language
        loader = PdfDocumentLoader(ocr_language="registry")
        score = lambda text: loader._score_ocr_candidate(psm="6", text=text)
        chinese = score("科學研究需要謹慎檢驗實驗結果和證據" * 10)
        english = score("Scientific research requires evidence and careful analysis. " * 10)
        self.assertEqual(horizontal_refinement_language([chinese] * 3 + [english], loader.ocr_language),
                         "eng+chi_sim+chi_tra")
        noise = score(chinese.text + "の한")
        self.assertIsNotNone(horizontal_refinement_language([noise] * 3 + [english], loader.ocr_language))
        for candidates in ([chinese, english, english, english], [english] * 4,
                           [chinese] * 3 + [score("研究には慎重な分析が必要です。" * 10)],
                           [chinese] * 3 + [score("과학 연구에는 증거가 필요합니다." * 10)]):
            self.assertIsNone(horizontal_refinement_language(candidates, loader.ocr_language))
        self.assertIsNone(horizontal_refinement_language([chinese] * 4, "chi_sim+chi_tra+eng"))

    def test_refinement_is_bounded_and_preserves_broad_candidates(self):
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        loader = PdfDocumentLoader(ocr_language="registry")
        broad = "科學研究需要謹慎檢驗實驗結果和證據" * 10
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                patch.object(loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, broad, "")) as run:
            text, candidates = loader._ocr_image_text_with_quality(
                image_path=Path("unused"), doc_name="fixture", page_index=0)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(run.call_count, 5)
        self.assertEqual(run.call_args.args[0][-4:], ["-l", "eng+chi_sim+chi_tra", "--psm", "6"])
        self.assertEqual(candidates[-1].psm, "6:zh")
        self.assertEqual(text, broad)
        self.assertEqual(loader.ocr_language, policy.DEFAULT_TESSERACT_OCR_LANGUAGE)

    def test_refinement_ranking_preserves_coverage_and_encoding_guards(self):
        from dataclasses import replace
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        loader = PdfDocumentLoader(ocr_language="registry")
        broad = loader._score_ocr_candidate(psm="11", text="科學研究需要謹慎檢驗實驗結果和證據" * 10)
        refined = replace(broad, psm="6:zh", passed=False, reasons=["fragmented_latin", "quality_rejected"],
                          score=broad.score + .01)
        self.assertEqual(loader._select_ocr_candidate([broad, broad, broad, refined]), refined)
        for rejected in (replace(refined, text=refined.text[:40]),
                         replace(refined, score=broad.score - .1),
                         replace(refined, reasons=["invalid_encoding"]),
                         replace(refined, cjk_ratio=.4)):
            self.assertEqual(loader._select_ocr_candidate([broad, broad, broad, rejected]), broad)


class OcrCapabilityGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.models = policy.required_ocr_models()
        self.data = b"pinned fixture model"
        for model in policy.VERTICAL_MODEL_HASHES:
            (self.directory / (model + ".traineddata")).write_bytes(self.data)
        for target, options in (
            ("scripts.verify_ocr_assets.shutil.which", {"return_value": "/bin/fixture"}),
            ("scripts.verify_ocr_assets.VERTICAL_MODEL_HASHES", {"new": dict.fromkeys(
                policy.VERTICAL_MODEL_HASHES, hashlib.sha256(self.data).hexdigest())}),
        ):
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def inventory(self, models=None):
        models = self.models if models is None else models
        return subprocess.CompletedProcess([], 0,
            f'List of available languages in "{self.directory}" ({len(models)}):\n' + "\n".join(models), "")

    def test_full_inventory_offline(self):
        with patch.object(assets.subprocess, "run", return_value=self.inventory()) as run:
            self.assertEqual(assets.verify_assets(), self.models)
            self.assertEqual(run.call_count, 1)

    def test_missing_model_fails_even_when_hint_english(self):
        with patch.dict(os.environ, {"DEEP_READER_PDF_OCR_LANGUAGE": "en"}), \
                patch.object(assets.subprocess, "run", return_value=self.inventory(tuple(m for m in self.models if m != "fra"))):
            with self.assertRaisesRegex(assets.OcrAssetError, "ocr_models_missing:fra"):
                assets.verify_assets()

    def test_corrupt_pinned_model_fails(self):
        (self.directory / "chi_tra_vert.traineddata").write_bytes(b"corrupt")
        with patch.object(assets.subprocess, "run", return_value=self.inventory()):
            with self.assertRaisesRegex(assets.OcrAssetError, "ocr_model_unverified:chi_tra_vert"):
                assets.verify_assets()

    def test_missing_engine_or_renderer_fails(self):
        with patch.object(assets.shutil, "which", side_effect=["/engine", None]):
            with self.assertRaisesRegex(assets.OcrAssetError, "ocr_engine_or_renderer_unavailable"):
                assets.verify_assets()

    def test_inventory_failure_and_unknown_directory_fail(self):
        for inventory in (subprocess.CompletedProcess([], 1, "", "private"),
                          subprocess.CompletedProcess([], 0, "\n".join(self.models), "")):
            with patch.object(assets.subprocess, "run", return_value=inventory):
                with self.assertRaises(assets.OcrAssetError):
                    assets.verify_assets()

    def test_build_loads_every_model_independently(self):
        with patch.object(assets.subprocess, "run", side_effect=[self.inventory()] + [
                subprocess.CompletedProcess([], 0, b"", b"") for _ in self.models]) as run:
            assets.verify_assets(load_models=True)
            loaded = [call.args[0][4] for call in run.call_args_list[1:]]
            self.assertEqual(tuple(loaded), self.models)

    def test_failed_model_load_even_zero_exit_is_rejected(self):
        for code, stderr in ((1, b"private"), (0, b"Failed loading language 'eng'")):
            with patch.object(assets.subprocess, "run", side_effect=[self.inventory(),
                    subprocess.CompletedProcess([], code, b"", stderr)]):
                with self.assertRaisesRegex(assets.OcrAssetError, "ocr_model_load_failed:eng"):
                    assets.verify_assets(load_models=True)

    def test_timeout_does_not_echo_process_details(self):
        with patch.object(assets.subprocess, "run", side_effect=subprocess.TimeoutExpired("private", 20)), \
                patch("sys.stderr", new_callable=io.StringIO) as output:
            self.assertEqual(assets.main([]), 1)
            self.assertNotIn("private", output.getvalue())

    def test_ocr_enabled_startup_fails_before_exec(self):
        with patch.dict(os.environ, {"DEEP_READER_PDF_OCR_ENABLED": "1"}), \
                patch.object(tokenizer_startup, "verify", return_value=0), \
                patch.object(tokenizer_startup, "verify_ocr", return_value=1), \
                patch.object(tokenizer_startup.os, "execvp") as execute, \
                patch("sys.argv", ["startup", "child"]):
            self.assertEqual(tokenizer_startup.main(), 1)
            execute.assert_not_called()

    def test_native_only_startup_does_not_require_ocr(self):
        with patch.dict(os.environ, {"DEEP_READER_PDF_OCR_ENABLED": "0"}), \
                patch.object(tokenizer_startup, "verify", return_value=0), \
                patch.object(tokenizer_startup, "verify_ocr") as verify_ocr, \
                patch.object(tokenizer_startup.os, "execvp") as execute, \
                patch("sys.argv", ["startup", "child"]):
            tokenizer_startup.main()
            verify_ocr.assert_not_called()
            execute.assert_called_once_with("child", ["child"])


if __name__ == "__main__":
    unittest.main()
