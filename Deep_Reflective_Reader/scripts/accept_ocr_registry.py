"""Real-engine horizontal OCR smoke; temporary generated PDFs, no user documents.

Run in the verified image with --font pointing to a mounted Unicode TTF font.
This checks representative newly supported scripts, not all-language accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unicodedata

from PIL import Image, ImageDraw, ImageFont

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from scripts.verify_ocr_assets import verify_assets


SAMPLES = {
    "fr": ("La recherche scientifique exige des preuves solides.",
           "Une analyse attentive permet de comprendre les observations."),
    "de": ("Wissenschaftliche Forschung braucht klare Beweise.",
           "Eine sorgsame Analyse hilft uns die Beobachtungen zu verstehen."),
    "ru": ("Научные исследования требуют доказательств.",
           "Внимательный анализ помогает понять результаты."),
    "ja": ("科学研究には証拠と慎重な分析が必要です。",
           "読書を通して新しい知識を学びます。"),
}


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).lower() if c.isalnum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    models = verify_assets(load_models=True)
    report = {"models_loaded": list(models), "cases": [], "limitations":
              "Four generated horizontal samples; not all-language or arbitrary-layout accuracy."}
    with tempfile.TemporaryDirectory(prefix="ocr-registry-") as temporary:
        root = Path(temporary)
        font = ImageFont.truetype(str(args.font), 36)
        for language, lines in SAMPLES.items():
            image = Image.new("RGB", (1400, 240), "white")
            draw = ImageDraw.Draw(image)
            for index, line in enumerate(lines):
                draw.text((40, 40 + index * 70), line, font=font, fill="black")
            pdf = root / (language + ".pdf")
            image.save(pdf, "PDF", resolution=150)
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            for hint in (language, "registry"):
                started = time.monotonic()
                loader = PdfDocumentLoader(root, ocr_language=hint)
                try:
                    result = loader.extract(language, mode="ocr")
                    # Consecutive reference characters survive transcription, not
                    # merely a nonempty exit; permit small font/OCR differences.
                    from difflib import SequenceMatcher
                    retention = SequenceMatcher(None, normalized("".join(lines)),
                                                normalized(result.text), autojunk=False).ratio()
                    case = {"language": language, "hint": hint, "retention": retention,
                            "passed": retention >= .90, "pages": len(result.pages)}
                except Exception as error:
                    case = {"language": language, "hint": hint, "passed": False,
                            "error_type": type(error).__name__}
                case.update(seconds=round(time.monotonic() - started, 3),
                            source_unchanged=source_hash == hashlib.sha256(pdf.read_bytes()).hexdigest())
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(c["passed"] and c["source_unchanged"] for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
