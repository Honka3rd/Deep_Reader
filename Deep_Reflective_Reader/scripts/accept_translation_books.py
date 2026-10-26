"""Explicit live-provider smoke on immutable book excerpts; no API/DB writes.

Run from the backend: python -m scripts.accept_translation_books --live
Load credentials with the repository's load_local_env.sh before starting Python.
Outputs stay in ignored data/test-books; semantic review is separate from assertions.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from scripts.private_book_corpus import CORPUS_ROOT, load_book, preserve
from section_tasks.translation_contracts import TranslationError
from section_tasks.translation_service import TranslationService


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.parse_args()
    from auth.openai_api_key_provider import OpenAIAPIKeyProvider
    from config.app_DI_config import AppDIConfig
    from llm.openai_llm_provider import OpenAILLMProvider

    fixtures = {}
    hashes = {}
    for case in ("fa794acef055f235", "085747b65094d52f"):
        root, snapshot, _ = load_book(case)
        fixtures[case] = snapshot.canonical_text
        manifest = json.loads((root / "manifest.json").read_text())
        for filename in (*manifest["files"], "manifest.json"):
            path = root / filename
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    english = fixtures["fa794acef055f235"]
    start = english.index("We were in class")
    english = english[start:english.index("\n\n", start)]
    chinese = fixtures["085747b65094d52f"]
    start = chinese.index("2017")
    chinese = chinese[start:chinese.index("\n\u3000", start)]
    cases = [("bovary-traditional", english, "en", "zh-Hant", "completed"),
             ("bovary-simplified", english, "en", "zh-Hans", "completed"),
             ("bovary-french", english, "en", "fr", "completed"),
             ("red-roulette-english", chinese, "zh", "en", "completed"),
             ("synthetic-unreadable-control", "@@@ ~~%%% \ufffd\ufffd ###", "zh", "en", "untranslatable")]
    config = AppDIConfig()
    provider = OpenAILLMProvider(OpenAIAPIKeyProvider(), model=config.llm_model,
                                target_max_output_tokens=config.target_max_output_tokens)
    service = TranslationService(provider)
    directory = CORPUS_ROOT / "translation" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    report = {"status": "failed", "model": str(config.llm_model), "cases": [],
              "semantic_review": "required", "source": "hash-verified private backups"}
    try:
        for name, text, source, target, expected in cases:
            result = service.translate(text=text, source_language=source, target_language=target)
            report["cases"].append({"case": name, "source_text": text, "result": asdict(result)})
            assert result.status == expected, name
            assert result.target_language == target
            if name == "red-roulette-english":
                assert "2017" in result.translated_text and "50" in result.translated_text
            print(f"PASS {name}: {result.status}", flush=True)
        report["status"] = "checks_passed"
    except TranslationError as error:
        report["error_code"] = error.code
        raise
    finally:
        report["backups_unchanged"] = all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in hashes.items())
        preserve(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode())
        print(f"Translation evidence: {directory}", flush=True)
        assert report["backups_unchanged"]


if __name__ == "__main__":
    run()
