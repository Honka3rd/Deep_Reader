"""Chinese-local OCR contracts; no remote model or private book required."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw

from doc_loaders.chinese_ocr_strategy import identity, recognize, runs, vertical_columns, marginal_components, read_margin, read_region, Column
import subprocess
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.document_load_errors import RawTextOcrFailedError
from doc_loaders.pdf_extraction_identity import identity_for_loader
from doc_loaders.chinese_ocr_strategy import MODEL_SHA256
from doc_loaders.pdf_extraction_contract import PdfExtractionError


class ChineseDirectionTests(unittest.TestCase):
    def margin_fixture(self):
        image = Image.new("L", (600, 900), 255)
        draw = ImageDraw.Draw(image)
        for y in range(30,150,30):
            draw.rectangle((20,y,40,y+20),fill=0)
        for y in range(180,650,10):
            draw.rectangle((29,y,31,y+2),fill=0)
        for y in range(690,780,30):
            draw.rectangle((25,y,35,y+18),fill=0)
        return image, Column(10,47,20,820,())

    def test_margin_geometry_requires_leader_not_just_edge_position(self):
        image,column = self.margin_fixture()
        self.assertIsNotNone(marginal_components(image,column))
        ImageDraw.Draw(image).rectangle((10,170,47,660),fill=255)
        self.assertIsNone(marginal_components(image,column))

    def test_margin_digits_and_title_preserved_with_component_evidence(self):
        image,column = self.margin_fixture()
        with patch("doc_loaders.chinese_ocr_strategy.read_region",
                   side_effect=[("新千年版序",94),("0",96),("0",95),("K)",0),("3",60)]):
            text,confidence,evidence = read_margin(None,image,column,None,language="chi_tra_vert",page_index=2)
        self.assertEqual(text,"新千年版序 003")
        self.assertEqual(confidence,60)
        self.assertEqual(len(evidence["components"]),4)
        self.assertEqual(evidence["region_role"],"margin_label")

    def test_unreadable_margin_is_not_silently_removed(self):
        image,column = self.margin_fixture()
        with patch("doc_loaders.chinese_ocr_strategy.read_region",return_value=("",0)):
            self.assertIsNone(read_margin(None,image,column,None,language="chi_tra_vert",page_index=2))

    def test_engine_failure_is_not_empty_quality_result(self):
        loader = PdfDocumentLoader(Path("."))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loader,"_run_process",return_value=subprocess.CompletedProcess([],1,"","private stderr")):
                with self.assertRaisesRegex(PdfExtractionError,"page 3") as caught:
                    read_region(loader,self.image(),Path(directory)/"crop.png",language="chi_tra_vert",psm="5",page_index=2)
        self.assertEqual(caught.exception.reason,"extraction_failed")
        self.assertNotIn("private",str(caught.exception))

    def image(self):
        image = Image.new("L", (600, 800), 255)
        draw = ImageDraw.Draw(image)
        for x in [180, 280, 380]:
            for y in range(100, 650, 45):
                draw.rectangle((x, y, x+25, y+25), outline=0, width=3)
        return image

    def test_column_order_is_right_to_left(self):
        columns = vertical_columns(self.image())
        self.assertEqual(len(columns), 3)
        self.assertEqual([c.left for c in columns], sorted((c.left for c in columns), reverse=True))

    def test_columns_stable_when_vertical_whitespace_changes(self):
        image = self.image()
        padded = Image.new("L",(image.width,image.height+500),255)
        padded.paste(image,(0,500))
        self.assertEqual([(c.left,c.right) for c in vertical_columns(image)],
                         [(c.left,c.right) for c in vertical_columns(padded)])

    def test_blank_has_no_vertical_columns(self):
        self.assertEqual(vertical_columns(Image.new("L", (500, 800), 255)), [])

    def test_ambiguous_sideways_fixture_rejects_implausible_transcription_size(self):
        with self.assertRaisesRegex(RawTextOcrFailedError,"region_quality_failed"):
            self.recognize_fixture(self.image().transpose(Image.Transpose.ROTATE_90))

    def test_horizontal_lines_do_not_trigger_vertical(self):
        image = Image.new("L", (600, 800), 255)
        draw = ImageDraw.Draw(image)
        for y in range(100, 650, 45):
            draw.rectangle((60,y,540,y+20), fill=0)
        self.assertEqual(vertical_columns(image), [])

    def test_non_chinese_is_not_routed(self):
        loader = PdfDocumentLoader(Path("."), ocr_language="eng")
        with patch("doc_loaders.chinese_ocr_strategy.Image.open", side_effect=AssertionError):
            self.assertIsNone(recognize(loader, "unread.png", doc_name="test", page_index=0))

    def test_missing_model_rejects_before_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"page.png"
            self.image().save(path)
            loader = PdfDocumentLoader(Path(directory), ocr_language="chi_tra")
            with patch("doc_loaders.chinese_ocr_strategy.identity", return_value={"sha256":None}):
                with self.assertRaisesRegex(PdfExtractionError,"model_missing_or_unverified"):
                    recognize(loader,path,doc_name="test",page_index=0)

    def test_unknown_model_has_no_false_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"TESSDATA_PREFIX":directory}):
                self.assertIsNone(identity()["sha256"])

    def recognize_fixture(self,image,text="中文直排辨識測試文字內容完整",confidence=90):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"page.png"
            image.save(path)
            loader = PdfDocumentLoader(Path(directory),ocr_language="chi_tra")
            with patch("doc_loaders.chinese_ocr_strategy.identity",return_value={"sha256":MODEL_SHA256}):
                with patch("doc_loaders.chinese_ocr_strategy.read_region",return_value=(text,confidence)):
                    return recognize(loader,path,doc_name="test",page_index=0)

    def test_latin_garbage_cannot_win_vertical_geometry(self):
        with self.assertRaisesRegex(RawTextOcrFailedError,"region_quality_failed"):
            self.recognize_fixture(self.image(),"AR BT Ad EKG OT HOE veer ER")

    def test_low_confidence_body_rejected(self):
        with self.assertRaisesRegex(RawTextOcrFailedError,"region_quality_failed"):
            self.recognize_fixture(self.image(),confidence=10)

    def test_column_mapping_and_quality_preserved(self):
        result = self.recognize_fixture(self.image())
        self.assertEqual(len(result.regions),3)
        for region in result.regions:
            self.assertEqual(result.text[region["char_start"]:region["char_end"]],"中文直排辨識測試文字內容完整")
            self.assertEqual(region["column_order"],"right_to_left")

    def test_uncovered_content_is_not_silently_dropped(self):
        image = self.image()
        ImageDraw.Draw(image).rectangle((20,100,140,600),fill=0)
        with self.assertRaisesRegex(RawTextOcrFailedError,"coverage_uncertain"):
            self.recognize_fixture(image)

    def test_separated_horizontal_header_and_vertical_body(self):
        image = Image.new("L",(600,1000),255)
        image.paste(self.image(),(0,200))
        ImageDraw.Draw(image).rectangle((60,30,540,55),fill=0)
        result = self.recognize_fixture(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.regions[0]["writing_mode"],"horizontal")
        self.assertEqual(result.regions[1]["writing_mode"],"vertical")

    def test_fast_identity_does_not_consult_chinese_models(self):
        loader = PdfDocumentLoader(Path("."))
        with patch("doc_loaders.chinese_ocr_strategy.identity",side_effect=AssertionError):
            self.assertEqual(identity_for_loader(loader,source_sha256="0"*64,mode="fast").mode,"fast")

    def test_ocr_identity_changes_with_direction_policy(self):
        loader = PdfDocumentLoader(Path("."))
        with patch("doc_loaders.chinese_ocr_strategy.identity",return_value={"policy":"one"}):
            first = identity_for_loader(loader,source_sha256="0"*64,mode="ocr")
        with patch("doc_loaders.chinese_ocr_strategy.identity",return_value={"policy":"two"}):
            second = identity_for_loader(loader,source_sha256="0"*64,mode="ocr")
        self.assertNotEqual(first,second)


if __name__ == "__main__":
    unittest.main()
