"""Offline startup gate shared by container and direct local API launches."""

import os
import sys

from scripts.prepare_tokenizer_assets import main as verify
from scripts.verify_ocr_assets import main as verify_ocr


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.tokenizer_startup <command> [args...]", file=sys.stderr)
        return 2
    result = verify(["--verify-only"])
    if result:
        return result
    if os.environ.get("DEEP_READER_PDF_OCR_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        result = verify_ocr(["--load-models"])
        if result:
            return result
    sys.stdout.flush()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
