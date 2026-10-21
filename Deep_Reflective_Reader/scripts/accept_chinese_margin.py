"""Real-engine regression against an immutable private Sample2 backup."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from doc_loaders.pdf_document_loader import PdfDocumentLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.corpus/"manifest.json").read_text())
    source = args.corpus/manifest["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    started = time.monotonic()
    result = PdfDocumentLoader(source.parent,ocr_enabled=True).extract(source.stem,mode="ocr")
    assert len(result.pages) == manifest["page_count"]
    assert list(result.empty_page_indexes) == manifest["blank_page_indexes"]
    assert result.page_quality[1]["blank_render_confirmed"]
    text = result.pages[2]
    assert manifest["page_3_body_phrase"] in re.sub(r"\s","",text)
    margins = [r for r in result.page_quality[2]["regions"] if r.get("region_role") == "margin_label"]
    assert len(margins) == 1
    margin = margins[0]
    assert text[margin["char_start"]:margin["char_end"]] == manifest["page_3_margin"]
    assert [c["role"] for c in margin["components"]] == ["title","page_digit","page_digit","page_digit"]
    for span in result.boundary_evidence.boundaries:
        assert result.text[span.char_start:span.char_end] == span.text
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    report = {"source_sha256":manifest["sha256"],"page_count":len(result.pages),
              "page_characters":[len(t) for t in result.pages],"margin":margin,
              "seconds":round(time.monotonic()-started,2),"passed":True,
              "limitations":manifest["scope"]}
    encoded = json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        args.output.write_text(encoded,encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
