"""Read-only real-engine acceptance against a hash-verified private backup."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import tempfile
from PIL import Image
from unittest.mock import patch
from doc_loaders.pdf_extraction_contract import PdfExtractionError

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.chinese_ocr_strategy import identity
from doc_loaders.chinese_ocr_strategy import recognize


def distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return row[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/test-books/ocr-direction-20261021"))
    parser.add_argument("--mode",choices=["ocr","auto"],default="ocr")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--mixed-fixture",action="store_true")
    parser.add_argument("--expect-native-coverage-rejection",action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.corpus/"manifest.json").read_text())
    source = args.corpus/manifest["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    loader = PdfDocumentLoader(source.parent,ocr_enabled=True)
    with patch.object(loader,"_run_process",side_effect=AssertionError("Fast invoked a subprocess")):
        native = loader.extract(source.stem,mode="fast")
    started = time.monotonic()
    try:
        result = loader.extract(source.stem, mode=args.mode)
    except PdfExtractionError as error:
        if not (args.mode == "auto" and args.expect_native_coverage_rejection
                and "native_coverage_unresolved" in str(error)):
            raise
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
        encoded = json.dumps({"mode":"auto","expected_rejection":"native_coverage_unresolved",
                              "source_unchanged":True,"fast_no_subprocess":True,
                              "persistence":"not invoked","model":identity()},indent=2)
        if args.output:
            args.output.write_text(encoded,encoding="utf-8")
        print(encoded)
        return
    assert not args.expect_native_coverage_rejection, "Expected rejection did not occur"
    assert len(result.pages) == 6
    assert "祖籍浙江" in re.sub(r"\s", "", result.pages[2])
    report = {"source_sha256":manifest["sha256"],"mode":args.mode,"model":identity(),
              "seconds":round(time.monotonic()-started,2),"pages":[],
              "fast_no_subprocess":True,"fast_characters":len(native.text),
              "limitations":["CER covers TOC column titles, not decorative headings or all punctuation.",
                             "Horizontal page is checked by a reference phrase, not a full transcription.",
                             "No LLM, profile generation, live API or Library writes."]}
    for index in [4,5]:
        expected = manifest["page_5_columns"] if index == 4 else list(zip(manifest["page_6_columns"],manifest["page_6_numbers"]))
        lines = [line.strip() for line in result.pages[index].splitlines() if re.search(r" \d{3}$",line.strip())]
        assert len(lines) == len(expected), (index+1, "column coverage",len(lines),len(expected))
        errors = size = 0
        for actual,(title,number) in zip(lines, expected):
            assert actual.endswith(" "+number), (index+1,"page association",number)
            a = "".join(re.findall(r"[\u3400-\u9fff]", actual))
            b = "".join(re.findall(r"[\u3400-\u9fff]", title))
            # Numbered headings may have a printed enumeration prefix.
            candidates = [a, re.sub(r"^[一二三四五六]+", "", a)]
            errors += min(distance(c,b) for c in candidates)
            size += len(b)
        cer = errors/max(1,size)
        assert cer <= manifest["max_cjk_character_error_rate"], (index+1, "transcription",cer)
        report["pages"].append({"page":index+1,"columns":len(lines),"cjk_error_rate":cer})
    if args.mixed_fixture:
        with tempfile.TemporaryDirectory() as directory, loader.preparation_scope():
            root = Path(directory)
            horizontal_path = loader._render_page_images(file_path=source,page_index=2,output_dir=root)[0]
            vertical_path = loader._render_page_images(file_path=source,page_index=4,output_dir=root)[0]
            with Image.open(horizontal_path) as im:
                header = im.convert("L").crop((660,730,1220,1100))
            with Image.open(vertical_path) as im:
                body = im.convert("L")
            mixed = Image.new("L",(body.width,body.height+500),255)
            mixed.paste(header,(100,20))
            mixed.paste(body,(0,500))
            path = root/"mixed.png"
            mixed.save(path)
            recognized = recognize(loader,path,doc_name="mixed-private-fixture",page_index=0)
            assert recognized is not None
            assert "祖籍浙江" in re.sub(r"\s","",recognized.text)
            assert len(re.findall(r" \d{3}$",recognized.text,re.MULTILINE)) == 12
            assert {r["writing_mode"] for r in recognized.regions} == {"horizontal","vertical"}
            report["mixed_header_vertical_body"] = "passed"
    encoded = json.dumps(report,ensure_ascii=False,indent=2)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    if args.output:
        args.output.write_text(encoded,encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
