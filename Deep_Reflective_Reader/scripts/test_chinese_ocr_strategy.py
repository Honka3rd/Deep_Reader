"""Chinese-local OCR contracts; no remote model or private book required."""

from pathlib import Path
from contextlib import ExitStack
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw

from doc_loaders.chinese_ocr_strategy import identity, recognize, runs, vertical_columns, marginal_components, read_margin, read_region, Column
import subprocess
from doc_loaders.pdf_execution_pass import PdfBudgetExceeded
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_identity import identity_for_loader
from doc_loaders.chinese_ocr_strategy import MODEL_SHA256, ChineseOcrFallback
from doc_loaders.pdf_extraction_contract import PdfExtractionError
from doc_loaders.ocr_review import bind_review_spans, page_review_regions
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from pypdf import PdfWriter
from scripts.test_pdf_page_inspection import add_image, page_fixture


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
            text,confidence,evidence = read_margin(None,image,column,None,language="chi_tra_vert",page_index=2)
        self.assertEqual(text, "")
        self.assertFalse(evidence["accepted"])
        self.assertIn("recognition_uncertain", evidence["quality_flags"])
        self.assertTrue(all(c["char_start"] == c["char_end"] == 0 for c in evidence["components"]))

    def test_uncertain_margin_preserves_title_and_every_digit_candidate(self):
        image,column = self.margin_fixture()
        with patch("doc_loaders.chinese_ocr_strategy.read_region",
                   side_effect=[(" 序 ",10),("0",96),("?",12),("",0),(" 3 ",60)]):
            text,confidence,evidence = read_margin(None,image,column,None,language="chi_tra_vert",page_index=2)
        self.assertEqual(text, " 序  0? 3 ")
        self.assertEqual(confidence, 10)
        self.assertFalse(evidence["accepted"])
        self.assertEqual(evidence["raw_candidate"], text)
        self.assertEqual([text[c["char_start"]:c["char_end"]] for c in evidence["components"]],
                         [" 序 ", "0", "?", " 3 "])

    def test_engine_failure_is_not_empty_quality_result(self):
        loader = PdfDocumentLoader(Path("."))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loader,"_run_process",return_value=subprocess.CompletedProcess([],1,"","private stderr")):
                with self.assertRaisesRegex(PdfExtractionError,"page 3") as caught:
                    read_region(loader,self.image(),Path(directory)/"crop.png",language="chi_tra_vert",psm="5",page_index=2)
        self.assertEqual(caught.exception.reason,"extraction_failed")
        self.assertNotIn("private",str(caught.exception))

    def test_margin_whitespace_retry_cannot_replace_meaningful_digit(self):
        image,column = self.margin_fixture()
        with patch("doc_loaders.chinese_ocr_strategy.read_region",
                   side_effect=[(" 序 ",90),("0",90),("0",90),(" 3 ",10),(" \t\n",0)]):
            text,_,evidence = read_margin(None,image,column,None,language="chi_tra_vert",page_index=0)
        self.assertEqual(text," 序  00 3 ")
        self.assertEqual([text[c["char_start"]:c["char_end"]] for c in evidence["components"]],
                         [" 序 ","0","0"," 3 "])
        self.assertFalse(evidence["accepted"])

    def test_margin_whitespace_only_components_are_empty_gaps(self):
        image,column = self.margin_fixture()
        for responses,expected in [([(" \t\n",0)]*7,""),
                                   ([(" \t\n",0),("0",90),("0",90),("3",90)],"003")]:
            with self.subTest(expected=expected), patch("doc_loaders.chinese_ocr_strategy.read_region",side_effect=responses):
                text,_,evidence = read_margin(None,image,column,None,language="chi_tra_vert",page_index=0)
            self.assertEqual(text,expected)
            self.assertEqual(evidence["components"][0]["char_start"],0)
            self.assertEqual(evidence["components"][0]["char_end"],0)
            self.assertFalse(evidence["accepted"])

    def test_tsv_preserves_token_whitespace_and_chinese_local_line_order(self):
        tsv = ("level\tblock_num\tpar_num\tline_num\tconf\ttext\n"
               "5\t1\t1\t1\t90\t 開頭 \n"
               "5\t1\t1\t1\t90\t \n"
               "5\t1\t1\t1\t90\t正文\n"
               "5\t1\t1\t2\t90\t末行 \n")
        loader = PdfDocumentLoader(Path("."))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loader,"_run_process",return_value=subprocess.CompletedProcess([],0,tsv,"")):
                horizontal,_ = read_region(loader,self.image(),Path(directory)/"crop.png",
                                           language="chi_tra",psm="6",page_index=0)
                vertical,_ = read_region(loader,self.image(),Path(directory)/"crop.png",
                                         language="chi_tra_vert",psm="5",page_index=0)
        self.assertEqual(horizontal," 開頭  正文\n末行 ")
        self.assertEqual(vertical," 開頭  正文末行 ")

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

    def test_ambiguous_sideways_fixture_retains_text_with_quality_diagnostics(self):
        result = self.recognize_fixture(self.image().transpose(Image.Transpose.ROTATE_90))
        self.assertTrue(result.text)
        self.assertTrue(any(not r["accepted"] for r in result.regions))
        for region in result.regions:
            if not region["accepted"]:
                self.assertEqual(result.text[region["char_start"]:region["char_end"]], region["raw_candidate"])

    def test_horizontal_lines_do_not_trigger_vertical(self):
        image = Image.new("L", (600, 800), 255)
        draw = ImageDraw.Draw(image)
        for y in range(100, 650, 45):
            draw.rectangle((60,y,540,y+20), fill=0)
        self.assertEqual(vertical_columns(image), [])
        self.assertIsNone(self.recognize_fixture(image))

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

    def recognize_fixture(self,image,text="中文直排辨識測試文字內容完整",confidence=90,*,allow_review=False,responses=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"page.png"
            image.save(path)
            loader = PdfDocumentLoader(Path(directory),ocr_language="chi_tra")
            with patch("doc_loaders.chinese_ocr_strategy.identity",return_value={"sha256":MODEL_SHA256}):
                with patch("doc_loaders.chinese_ocr_strategy.read_region",return_value=(text,confidence),side_effect=responses):
                    return recognize(loader,path,doc_name="test",page_index=0,allow_review=allow_review)

    def test_latin_garbage_is_retained_and_flagged(self):
        text = "AR BT Ad EKG OT HOE veer ER"
        result = self.recognize_fixture(self.image(), text)
        self.assertEqual(result.text, "\n".join([text]*3))
        self.assertTrue(all(not r["accepted"] for r in result.regions))

    def test_low_confidence_body_retained_in_both_review_modes(self):
        text = "  中文直排辨識測試\t文字內容完整  \n"
        for allow_review in (False, True):
            with self.subTest(allow_review=allow_review):
                result = self.recognize_fixture(self.image(),text,confidence=10,allow_review=allow_review)
                self.assertEqual(result.text, "\n".join([text]*3))
                for region in result.regions:
                    self.assertFalse(region["accepted"])
                    self.assertEqual(result.text[region["char_start"]:region["char_end"]],text)
                    self.assertEqual(region["raw_candidate"],text)
                    self.assertIn("recognition_uncertain",region["quality_flags"])

    def test_mixed_quality_columns_preserve_order_and_exact_uncertain_span(self):
        texts = ["第一列中文直排辨識文字內容完整", " ? \t\ufffd ", "第三列中文直排辨識文字內容完整"]
        result = self.recognize_fixture(self.image(),responses=list(zip(texts,[90,0,95])))
        self.assertEqual(result.text,"\n".join(texts))
        self.assertEqual([r["accepted"] for r in result.regions],[True,False,True])
        self.assertEqual([result.text[r["char_start"]:r["char_end"]] for r in result.regions],texts)
        self.assertEqual(result.regions[1]["char_start"],len(texts[0])+1)
        self.assertEqual(result.regions[1]["char_end"],len(texts[0])+1+len(texts[1]))

    def test_empty_regions_do_not_invent_text_or_separators(self):
        for responses in [[("",0)]*3, [("",0),("中文",10),("",0)],
                          [(" \t",0),(" 中文 \n",10),("\n ",0)]]:
            with self.subTest(responses=responses):
                result = self.recognize_fixture(self.image(),responses=responses)
                self.assertEqual(result.text,"\n".join(t for t,c in responses if t.strip()))
                for region,(text,_) in zip(result.regions,responses):
                    self.assertLessEqual(region["char_end"],len(result.text))
                    self.assertEqual(result.text[region["char_start"]:region["char_end"]],text if text.strip() else "")

    def test_whitespace_only_candidates_are_empty_gaps(self):
        text = " \t\n  "
        result = self.recognize_fixture(self.image(),text,confidence=0)
        self.assertEqual(result.text,"")
        self.assertTrue(all(r["char_start"] == r["char_end"] == 0 for r in result.regions))

    def test_uncertain_numeric_label_keeps_whitespace_and_nonempty_first_candidate(self):
        image = Image.new("L",(600,1000),255)
        column = Column(380,420,100,950,((100,500),(850,950)))
        body = "中文直排辨識測試文字"
        for label in (" 0 ? 3 \n", "", " \t\n"):
            with self.subTest(label=label), patch("doc_loaders.chinese_ocr_strategy.vertical_columns",return_value=[column]):
                result = self.recognize_fixture(image,responses=[(label,10),(" \t\n",0),(body,90)])
            label = label if label.strip() else ""
            self.assertEqual(result.text,body+(" "+label if label else ""))
            region = result.regions[0]
            self.assertFalse(region["accepted"])
            self.assertIn("chinese_column_label_uncertain",region["quality_flags"])
            self.assertEqual(result.text[region["char_start"]:region["char_end"]],region["raw_candidate"])
            component = region["components"][1]
            self.assertEqual(result.text[component["char_start"]:component["char_end"]],label)

    def test_nonempty_short_label_survives_empty_glyph_retry_and_page_confidence(self):
        image = Image.new("L",(600,800),255)
        ImageDraw.Draw(image).rectangle((100,100,120,120),fill=0)
        columns = [Column(300,340,100,600,((100,600),)),Column(90,130,90,180,((90,180),))]
        body = "中文直排辨識測試文字"
        with patch("doc_loaders.chinese_ocr_strategy.vertical_columns",return_value=columns):
            result = self.recognize_fixture(image,responses=[(body,90),("目錄",10),("",0)])
        self.assertEqual(result.text,body+"\n目錄")
        self.assertFalse(result.regions[1]["accepted"])
        self.assertIn("low_confidence_short_label",result.regions[1]["quality_flags"])

    def test_execution_failure_after_recognized_column_remains_explicit(self):
        for failure in (subprocess.TimeoutExpired("ocr",1),PdfBudgetExceeded("ocr budget")):
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)):
                    self.recognize_fixture(self.image(),responses=[("中文",90),failure])

    def test_short_label_survives_whitespace_only_or_partial_glyph_replacement(self):
        image = Image.new("L",(600,800),255)
        draw = ImageDraw.Draw(image)
        for y in (100,140):
            draw.rectangle((100,y,120,y+20),fill=0)
        columns = [Column(300,340,100,600,((100,600),)),Column(90,130,90,180,((100,120),(140,160)))]
        for parts in [[(" \t",0),("\n ",0)],[("目",90),(" \t",0)]]:
            with self.subTest(parts=parts), patch("doc_loaders.chinese_ocr_strategy.vertical_columns",return_value=columns):
                result = self.recognize_fixture(image,responses=[("正文",90),("目錄",10),*parts])
            self.assertEqual(result.text,"正文\n目錄")
            region = result.regions[-1]
            self.assertEqual(result.text[region["char_start"]:region["char_end"]],"目錄")
            self.assertFalse(region["accepted"])

    def test_column_mapping_and_quality_preserved(self):
        result = self.recognize_fixture(self.image())
        self.assertEqual(len(result.regions),3)
        for region in result.regions:
            self.assertEqual(result.text[region["char_start"]:region["char_end"]],"中文直排辨識測試文字內容完整")
            self.assertEqual(region["column_order"],"right_to_left")

    def test_uncovered_content_defers_to_bounded_full_image_candidates(self):
        image = self.image()
        ImageDraw.Draw(image).rectangle((20,100,140,600),fill=0)
        with patch("doc_loaders.chinese_ocr_strategy.read_margin",side_effect=AssertionError("partial OCR")):
            self.assertEqual(self.recognize_fixture(image),
                             ChineseOcrFallback("chinese_layout_coverage_uncertain",(0,0,600,800)))

    def test_excessive_columns_defer_without_unbounded_region_recognition(self):
        columns = [Column(100,140,100,600,((100,600),))]*65
        with patch("doc_loaders.chinese_ocr_strategy.vertical_columns",return_value=columns):
            with patch("doc_loaders.chinese_ocr_strategy.read_margin",side_effect=AssertionError("partial OCR")):
                self.assertEqual(self.recognize_fixture(Image.new("L",(600,800),255)),
                                 ChineseOcrFallback("chinese_layout_coverage_uncertain",(0,0,600,800)))

    def test_mixed_uncertain_geometry_retains_header_and_fallback_band_text(self):
        image = Image.new("L",(600,1000),255)
        image.paste(self.image(),(0,200))
        draw = ImageDraw.Draw(image)
        draw.rectangle((60,30,540,55),fill=0)
        draw.rectangle((20,300,140,800),fill=0)
        texts = ["Header", "保留所有識別文字，即使版面不確定。"]
        result = self.recognize_fixture(image,responses=[(text,90) for text in texts])
        self.assertEqual(result.text,"\n".join(texts))
        self.assertFalse(result.regions[1]["accepted"])
        self.assertIn("chinese_layout_coverage_uncertain",result.regions[1]["quality_flags"])
        for region,text in zip(result.regions,texts):
            self.assertEqual(result.text[region["char_start"]:region["char_end"]],text)

    def test_separated_horizontal_header_and_vertical_body(self):
        image = Image.new("L",(600,1000),255)
        image.paste(self.image(),(0,200))
        ImageDraw.Draw(image).rectangle((60,30,540,55),fill=0)
        result = self.recognize_fixture(image)
        self.assertIsNotNone(result)
        self.assertEqual(result.regions[0]["writing_mode"],"horizontal")
        self.assertEqual(result.regions[1]["writing_mode"],"vertical")

    def test_mixed_band_uncertainty_preserves_all_text_and_translates_spans(self):
        image = Image.new("L",(600,1000),255)
        image.paste(self.image(),(0,200))
        ImageDraw.Draw(image).rectangle((60,30,540,55),fill=0)
        texts = [" \t???\n ","第一列中文直排辨識文字內容完整","錯 ? ","第三列中文直排辨識文字內容完整"]
        for allow_review in (False,True):
            with self.subTest(allow_review=allow_review):
                result = self.recognize_fixture(image,allow_review=allow_review,responses=list(zip(texts,[90,90,10,90])))
                self.assertEqual(result.text,"\n".join(texts))
                self.assertEqual([r["writing_mode"] for r in result.regions],
                                 ["horizontal","vertical","vertical","vertical"])
                self.assertEqual([r["accepted"] for r in result.regions],[False,True,False,True])
                for region,text in zip(result.regions,texts):
                    self.assertEqual(result.text[region["char_start"]:region["char_end"]],text)
                    if not region["accepted"]:
                        self.assertEqual(region["raw_candidate"],text)
                self.assertGreater(result.regions[1]["bbox"][1],200)

    def test_empty_mixed_bands_have_no_invented_separator_or_out_of_bounds_span(self):
        image = Image.new("L",(600,1000),255)
        image.paste(self.image(),(0,200))
        ImageDraw.Draw(image).rectangle((60,30,540,55),fill=0)
        for texts in (["", "中文", "", ""], ["題頭", "", "", ""]):
            with self.subTest(texts=texts):
                result = self.recognize_fixture(image,responses=[(text,10) for text in texts])
                self.assertEqual(result.text,"\n".join(text for text in texts if text))
                for region,text in zip(result.regions,texts):
                    self.assertLessEqual(region["char_end"],len(result.text))
                    self.assertEqual(result.text[region["char_start"]:region["char_end"]],text)

    def test_mixed_margin_component_spans_and_boxes_are_page_relative(self):
        margin,_ = self.margin_fixture()
        image = Image.new("L",(600,2000),255)
        image.paste(margin,(0,200))
        ImageDraw.Draw(image).rectangle((60,30,540,55),fill=0)

        def columns(part):
            return [Column(10,47,0,part.height,())] if part.height > 200 else []

        with patch("doc_loaders.chinese_ocr_strategy.vertical_columns",side_effect=columns):
            result = self.recognize_fixture(image,responses=[("Header",90),(" 序 ",10),
                                                            ("0",90),("?",10),("",0),("3",90)])
        self.assertEqual(result.text,"Header\n 序  0?3")
        region = result.regions[1]
        self.assertFalse(region["accepted"])
        self.assertEqual(result.text[region["char_start"]:region["char_end"]]," 序  0?3")
        self.assertEqual([result.text[c["char_start"]:c["char_end"]] for c in region["components"]],
                         [" 序 ","0","?","3"])
        self.assertTrue(all(c["bbox"][1] >= region["bbox"][1] for c in region["components"]))
        self.assertGreater(region["leader_bbox"][1],region["bbox"][1])

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

    def whitespace_fixture(self):
        texts = [" \t第一列中文直排辨識文字內容完整  ",
                 "第二列中文直排辨識文字內容完整",
                 "  第三列中文直排辨識文字內容完整\t\n "]
        result = self.recognize_fixture(self.image(),responses=list(zip(texts,[10,90,10])))
        self.assertEqual(result.text,"\n".join(texts))
        return result, [texts[0],texts[2]]

    def assert_review_spans(self, result, expected_texts):
        records = result.provenance["source_review"]["regions"]
        self.assertEqual([r["raw_candidate"] for r in records],expected_texts)
        digest = hashlib.sha256(result.text.encode()).hexdigest()
        for record in records:
            self.assertEqual(record["body_inclusion"],"included")
            span = record["canonical_span"]
            self.assertEqual(span["text_sha256"],digest)
            self.assertEqual(result.text[span["char_start"]:span["char_end"]],record["raw_candidate"])
        for boundary in result.boundary_evidence.boundaries:
            self.assertEqual(result.text[boundary.char_start:boundary.char_end],boundary.text)

    def test_review_helper_binds_meaningful_region_whitespace_exactly(self):
        directed,expected = self.whitespace_fixture()
        prefix = " native page \n"
        canonical = prefix+"\n"+directed.text
        offset = len(prefix)+1
        boundary = PdfPageTextBoundary(page_index=2,char_start=offset,
                                       char_end=offset+len(directed.text),text=directed.text)
        records = page_review_regions(2,directed.text,
                                      {"selected_passed":True,"regions":directed.regions},
                                      [0,0,600,800],200)
        self.assertEqual([r["raw_candidate"] for r in records],expected)
        bound = bind_review_spans(records,canonical,[boundary])
        for record,source in zip(bound,[directed.regions[0],directed.regions[2]]):
            span = record["canonical_span"]
            self.assertEqual(span["char_start"],offset+source["char_start"])
            self.assertEqual(span["char_end"],offset+source["char_end"])
            self.assertEqual(canonical[span["char_start"]:span["char_end"]],record["raw_candidate"])

    def extract_whitespace_fixture(self,mode,*,native_prefix=False,empty_columns=False):
        directed,expected = self.whitespace_fixture()
        if empty_columns:
            text = " \t保留非空中文區域兩端的原始空白\n "
            directed = self.recognize_fixture(self.image(),responses=[("",0),(text,10),("",0)])
            expected = [text]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            writer = PdfWriter()
            native = "  Original native paragraph with enough readable words.  "
            kinds = ["native" if native_prefix else "scan","blank","scan","blank"]
            for kind in kinds:
                content = (f"BT /F1 3 Tf 10 60 Td ({native}) Tj ET".encode() if kind == "native"
                           else b"q 100 0 0 100 0 0 cm /Im Do Q" if kind == "scan" else b"")
                _,page = page_fixture(content,with_font=kind=="native")
                if kind == "scan":
                    add_image(page)
                writer.add_page(page)
            writer.write(root/"sample.pdf")
            loader = PdfDocumentLoader(root,ocr_enabled=True,ocr_language="chi_tra")

            def render(*,page_index,output_dir,**kwargs):
                path = output_dir/f"page-{page_index}.png"
                image = Image.new("L",(600,800),255) if kinds[page_index] == "blank" else self.image()
                image.save(path)
                return [path]

            stack.enter_context(patch("doc_loaders.pdf_document_loader.shutil.which",return_value=str(Path(__file__).resolve())))
            stack.enter_context(patch.object(loader,"_binary_identity",side_effect=lambda cmd:(cmd,"fixture-v1")))
            stack.enter_context(patch.object(loader,"_tesseract_version",return_value="fixture-v1"))
            stack.enter_context(patch.object(loader,"_run_process",
                                return_value=subprocess.CompletedProcess([],0,"chi_tra\n","")))
            stack.enter_context(patch.object(loader,"_render_page_images",side_effect=render))
            stack.enter_context(patch("doc_loaders.chinese_ocr_strategy.recognize",return_value=directed))
            result = loader.extract("sample",mode=mode)
            session = loader.open_source("sample",mode=mode)
            artifact = session.extract()
            self.assertIs(session.extract(),artifact)
            self.assertEqual(artifact.result.canonical_text,result.text)
            self.assertEqual(artifact.result.text_sha256,hashlib.sha256(result.text.encode()).hexdigest())
            self.assertEqual(artifact.storage_metadata["source_review"],result.provenance["source_review"])
            self.assertEqual(artifact.page_boundaries,tuple(result.boundary_evidence.boundaries))
            for span in artifact.ocr_spans:
                self.assertEqual(artifact.result.canonical_text[span.char_start:span.char_end],
                                 result.pages[span.page_index])
            session.verify_source()
        expected_pages = (native if native_prefix else directed.text,"",directed.text,"")
        self.assertEqual(result.pages,expected_pages)
        separator = "\n\n" if mode == "auto" and not native_prefix else "\n"
        self.assertEqual(result.text,separator.join(text for text in expected_pages if text))
        return result,expected if native_prefix else expected*2

    def test_full_ocr_adapter_binds_region_whitespace_across_blank_pages(self):
        result,expected = self.extract_whitespace_fixture("ocr")
        self.assert_review_spans(result,expected)

    def test_auto_adapter_binds_region_whitespace_for_scan_and_native_joins(self):
        for native_prefix in (False,True):
            with self.subTest(native_prefix=native_prefix):
                result,expected = self.extract_whitespace_fixture("auto",native_prefix=native_prefix)
                self.assert_review_spans(result,expected)

    def test_empty_outer_columns_do_not_shift_nonempty_region_bindings(self):
        for mode in ("ocr","auto"):
            with self.subTest(mode=mode):
                result,expected = self.extract_whitespace_fixture(mode,empty_columns=True)
                records = result.provenance["source_review"]["regions"]
                mapped = [r for r in records if "canonical_span" in r]
                self.assertEqual([r["raw_candidate"] for r in mapped],expected)
                for record in mapped:
                    span = record["canonical_span"]
                    self.assertEqual(result.text[span["char_start"]:span["char_end"]],record["raw_candidate"])
                for record in records:
                    if not record["raw_candidate"]:
                        self.assertNotIn("canonical_span",record)


if __name__ == "__main__":
    unittest.main()
