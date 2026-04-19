#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [[ ! -f "data/raw/APPLE.pdf" ]]; then
  echo "FAIL: data/raw/APPLE.pdf not found."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import re

from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.text_document_loader import TextDocumentLoader


factory = DocumentLoaderFactory()

cases = [
    ("APPLE", PdfDocumentLoader),
    ("APPLE.pdf", PdfDocumentLoader),
    ("Madame Bovary", TextDocumentLoader),
]

loaded_texts: dict[str, str] = {}

for doc_name, expected_loader_type in cases:
    loader = factory.get(doc_name)
    if not isinstance(loader, expected_loader_type):
        raise AssertionError(
            f"{doc_name}: expected {expected_loader_type.__name__}, got {loader.__class__.__name__}"
        )

    text = loader.load(doc_name)
    if not text.strip():
        raise AssertionError(f"{doc_name}: loaded text is empty")

    loaded_texts[doc_name] = text
    print(
        f"[OK] {doc_name}: loader={loader.__class__.__name__}, "
        f"chars={len(text)}"
    )

apple_text = loaded_texts["APPLE"].upper()
if re.search(r"FORM\s*10\s*-\s*K", apple_text) is None:
    raise AssertionError("APPLE: expected annual report marker 'FORM 10-K' not found")

print("[PASS] Apple annual report PDF loader test passed.")
PY

