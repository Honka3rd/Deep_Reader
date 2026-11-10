"""Offline OCR capability gate; acquisition belongs exclusively to image setup."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from doc_loaders.pdf_ocr_language_policy import (
    VERTICAL_MODEL_HASHES, normalize_tesseract_language_config, required_ocr_models,
)


class OcrAssetError(RuntimeError):
    pass


def verify_assets(*, load_models=False):
    engine = os.environ.get("DEEP_READER_TESSERACT_CMD", "tesseract").strip() or "tesseract"
    renderer = os.environ.get("DEEP_READER_PDF_RENDERER_CMD", "pdftoppm").strip() or "pdftoppm"
    # A narrowed hint does not relax capability completeness.
    normalize_tesseract_language_config(os.environ.get("DEEP_READER_PDF_OCR_LANGUAGE"))
    required = required_ocr_models()
    if not shutil.which(engine) or not shutil.which(renderer):
        raise OcrAssetError("ocr_engine_or_renderer_unavailable")
    result = subprocess.run([engine, "--list-langs"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise OcrAssetError("ocr_model_inventory_failed")
    lines = result.stdout.splitlines()
    missing = set(required) - {line.strip() for line in lines}
    if missing:
        raise OcrAssetError("ocr_models_missing:" + ",".join(sorted(missing)))
    header = re.search(r'^List of available languages in "(.+)" \(\d+\):$', result.stdout, re.MULTILINE)
    if header is None:
        raise OcrAssetError("ocr_model_directory_unresolved")
    directory = Path(header.group(1))
    for model, expected in VERTICAL_MODEL_HASHES.items():
        if model not in required:
            continue
        path = directory / (model + ".traineddata")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise OcrAssetError("ocr_model_unverified:" + model)
    if load_models:
        # Each model is loaded independently: Tesseract may otherwise continue
        # successfully when only a subset of a combined -l selection is usable.
        blank = b"P5\n32 32\n255\n" + b"\xff" * 1024
        for model in required:
            loaded = subprocess.run(
                [engine, "stdin", "stdout", "-l", model, "--psm", "6"],
                input=blank, capture_output=True, timeout=30,
            )
            if loaded.returncode or b"Failed loading language" in loaded.stderr:
                raise OcrAssetError("ocr_model_load_failed:" + model)
    return required


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true", help="Build-time per-model engine check")
    args = parser.parse_args(argv)
    try:
        models = verify_assets(load_models=args.load_models)
    except (OcrAssetError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        # Do not emit subprocess stderr, arbitrary paths or environment values.
        detail = str(error) if isinstance(error, (OcrAssetError, ValueError)) else type(error).__name__
        print("OCR capability verification failed: " + detail, file=sys.stderr)
        return 1
    print(f"OCR capabilities verified offline: {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
