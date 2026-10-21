"""Real backup bytes through cold/warm preparation with isolated stores, no LLM."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accept_summary_note_books import BOOKS, CORPUS, ROOT, encoded, verify_backup
from private_book_corpus import preserve
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _fake_document
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from document_structure.structured_document_builder import StructuredDocumentBuilder

# Accepted canonical digests from common-language-20261021T094245.json.
EXPECTED = {
    "Madame Bovary": ("en", 652001, "0c90228bd5eda79be9a16706e961b3b50ee2cf58720e4e7466741761e3780f33"),
    "红色赌盘_261014_145614": ("zh", 147401, "ff9b7b1ce6b1684db85e4cf4ba8711f782f071a435ffd64c14642ebb44670ffb"),
}


def main():
    verify_backup()
    factory = DocumentLoaderFactory(CORPUS / "raw")
    reports = []
    for filename in BOOKS:
        name = Path(filename).stem
        language, length, digest = EXPECTED[name]
        profile = ROOT / "data/faiss" / name / "profile.json"
        before = profile.read_bytes() if profile.exists() else None
        loader = factory.get(name)
        store = _StructuredStore(exists_value=False, document=_fake_document(name))
        pipeline = _build_pipeline(loader=loader, store=store)
        pipeline.loader_factory = factory
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        pipeline.language_detector.detect = lambda text: language
        with patch.object(loader, "extract", wraps=loader.extract) as extract:
            cold = pipeline.prepare(name, mode="base", pdf_extraction_mode="fast")
            assert cold.structured_document_ready, cold.errors
            assert len(cold.raw_text) == length
            assert hashlib.sha256(cold.raw_text.encode()).hexdigest() == digest
            sections = [s for c in store.document.chapters for s in c.sections]
            assert "".join(s.content for s in sections) == cold.raw_text
            extract.assert_called_once()
            with patch.object(pipeline, "_prepare_profile", side_effect=AssertionError("warm profile")):
                warm = pipeline.prepare(name, mode="base", pdf_extraction_mode="fast")
            assert warm.structured_document_ready, warm.errors
            extract.assert_called_once()
            assert store.save_calls == 1
        assert (profile.read_bytes() if profile.exists() else None) == before
        reports.append({"book": name, "source_chars": length, "source_sha256": digest,
            "exact_text_preserved": True, "extractions": 1, "warm_extractions": 0,
            "warm_profile_calls": 0, "library_profile_unchanged": True,
            "chapters": len(store.document.chapters), "sections": len(sections),
            "storage": "isolated_in_memory", "live_llm": False})
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    verify_backup()
    path = CORPUS / ("loader-boundary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    preserve(path, encoded(reports))
    print(f"PASS report={path}")


if __name__ == "__main__":
    main()
