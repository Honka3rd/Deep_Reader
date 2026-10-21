"""Read-only Common acceptance after the explicit two-book reset harness."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.accept_summary_note_books import BOOKS, CORPUS, ROOT, encoded, verify_backup
from scripts.private_book_corpus import preserve
from document_structure.common_parse_policy import COMMON_POLICY_VERSION
from document_structure.section_splitter import CommonSectionSplitter
from language.language_code import LanguageCode


def main():
    verify_backup()
    import main as application
    coordinator = application.section_task_coordinator
    results = []
    for filename in BOOKS:
        name = Path(filename).stem
        source_bytes = (ROOT / "data/raw" / filename).read_bytes()
        assert source_bytes == (CORPUS / "raw" / filename).read_bytes()
        profile_path = ROOT / "data/faiss" / name / "profile.json"
        profile_before = profile_path.read_bytes()
        document = coordinator.document_artifact_repository.load_document(name)
        source = coordinator.read_enhancement_state(name).source
        text = source.canonical_text
        sections = [s for chapter in document.chapters for s in chapter.sections]
        assert "".join(s.content for s in sections) == text
        policy = document.parse_provenance["common_parse_policy"]
        assert policy["version"] == COMMON_POLICY_VERSION
        language = LanguageCode(document.language)
        reparsed = CommonSectionSplitter().split(text, language,
            toc_hint=policy["toc_hint"])
        assert "".join(s.content for s in reparsed) == text
        assert profile_path.read_bytes() == profile_before
        results.append({"book": name, "language": language.value,
            "policy": policy["version"], "source_chars": len(text),
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "backup_bytes_match": True, "exact_source_preserved": True,
            "profile_unchanged": True, "chapters": len(document.chapters),
            "sections": len(sections), "titled_common_sections": sum(s.title is not None for s in reparsed)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    verify_backup()
    output = CORPUS / ("common-language-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    preserve(output, encoded(results))
    print(f"PASS report={output}")


if __name__ == "__main__":
    main()
