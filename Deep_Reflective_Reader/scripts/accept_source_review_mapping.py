"""SRMAP-07 backup-only file/PostgreSQL acceptance, without OCR/model/API calls.

Run from the backend root with PYTHONPATH=.:scripts and a disposable test DSN.
Natural review evidence and constructed real-page evidence are reported separately.
All writes use temporary storage or a new private report; backups are never edited.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from document_structure.ocr_cleanup_overlay import canonical_text, block_range, unit_ranges
from document_structure.reading_content import ReadingContentService, original_blocks
from document_structure.source_review_blocks import region_block_mapping, persist_region_mapping
from document_structure.structured_document import StructuredDocument
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts.private_book_corpus import CORPUS_ROOT, load_book, preserve
from section_tasks.note_interaction_service import NoteInteractionService, NoteConflict
from shared.task_unit_model import TaskUnit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_cases():
    directory, snapshot, document = load_book("7270c4d2c581f525")
    cases = [("7270c4d2c581f525", directory / "source.pdf", document)]
    paths = [directory / name for name in (
        "manifest.json", "source.pdf", "snapshot.json", "original-document.json")]
    assert len(canonical_text(document)) == snapshot.text_length
    for name in ("sample", "sample2", "noise"):
        directory = CORPUS_ROOT / "cleanup-20261023" / name
        source = directory / "book.pdf"
        metadata = json.loads((directory / "manifest.json").read_text())
        receipt = json.loads((directory / "extraction.json").read_text())
        encoded = (directory / "document.json").read_bytes()
        assert sha(source.read_bytes()) == metadata["source_sha256"], "source_backup_hash_mismatch"
        assert sha(encoded) == receipt["document_sha256"], "document_backup_hash_mismatch"
        document = StructuredDocument.from_dict(json.loads(encoded))
        extraction = document.parse_provenance["extraction"]
        assert extraction["identity"]["source_sha256"] == metadata["source_sha256"]
        assert sha(canonical_text(document).encode()) == extraction["text_sha256"]
        assert len(canonical_text(document)) == extraction["text_length"]
        cases.append((name, source, document))
        paths.extend(directory / name for name in ("book.pdf", "manifest.json", "extraction.json", "document.json"))
    return cases, {path: sha(path.read_bytes()) for path in paths}


def materialize_fixture(document):
    """Exact section units for fixtures lacking layout; no parsing or extraction."""
    return replace(document, chapters=[replace(chapter, sections=[replace(section,
        task_units=section.task_units or [TaskUnit(
            unit_id=f"srmap-{chapter.chapter_id}-{section.section_id}", title=None,
            container_title=None, content=section.content, source_section_ids=[section.section_id],
            is_fallback_generated=False, parent_section_id=section.section_id)])
        for section in chapter.sections]) for chapter in document.chapters])


def natural_evidence(document):
    before = deepcopy(document.to_dict())
    mappings = region_block_mapping(document)
    regions = document.parse_provenance.get("source_review", {}).get("regions", [])
    decisions = document.parse_provenance.get("reading_content", {}).get("regions", {})
    text = canonical_text(document)
    rows = []
    for region in regions:
        quote = region.get("raw_candidate", "")
        pages = [p for p in document.parse_provenance.get("source_page_spans", [])
                 if p["page_index"] == region["page_index"]]
        rows.append({"region_id": region["id"], "page_index": region["page_index"],
            "candidate_length": len(quote), "candidate_sha256": sha(quote.encode()),
            "candidate_in_canonical": bool(quote) and quote in text,
            "original_decision": decisions.get(region["id"]), "pages": pages,
            **mappings[region["id"]]})
    assert document.to_dict() == before, "mapping_read_mutated_input"
    return {"canonical_chars": len(text), "canonical_sha256": sha(text.encode()),
        "original_region_count": len(regions),
        "body_status_counts": dict(Counter(row["body_status"] for row in rows)),
        "mapping_status_counts": dict(Counter(row["mapping_status"] for row in rows)), "regions": rows}


def real_page_case(document):
    """Create review evidence from a hash-verified accepted page, without canonical_span.

    This does not claim the loader originally flagged that page. A whole-page quote
    and a strict subquote exercise shared links and partial block intersection.
    """
    missing_units = any(not s.task_units for c in document.chapters for s in c.sections)
    document = materialize_fixture(document)
    text = canonical_text(document)
    for page in document.parse_provenance["source_page_spans"]:
        start, end = page["char_start"], page["char_end"]
        if end - start < 30:
            continue
        provenance = deepcopy(document.parse_provenance)
        provenance.pop("source_review_mapping", None)
        provenance.pop("reading_content", None)
        provenance["source_review"]["regions"] = [
            {"id": "real-page", "page_index": page["page_index"], "raw_candidate": text[start:end]},
            {"id": "real-subquote", "page_index": page["page_index"], "raw_candidate": text[start+3:start+23]}]
        candidate = replace(document, parse_provenance=provenance)
        mappings = region_block_mapping(candidate)
        if all(m["body_status"] == "linked" for m in mappings.values()):
            return candidate, {"kind": "constructed_review_from_accepted_real_page",
                "natural_loader_region": False, "canonical_span_injected": False,
                "page_index": page["page_index"], "page_char_start": start, "page_char_end": end,
                "page_text_sha256": sha(text[start:end].encode()),
                "subquote_char_start": start+3, "subquote_char_end": start+23,
                "materialized_missing_units": missing_units}
    raise AssertionError("no_verified_real_page_can_supply_linked_fixture")


def correspondence(document):
    text = canonical_text(document)
    blocks = {}
    for chapter, section, unit, block in original_blocks(document):
        span = block_range(unit, block, unit_ranges(section)[unit.unit_id])
        blocks[block.block_id] = {"chapter_id": chapter.chapter_id, "section_id": section.section_id,
            "task_unit_id": unit.unit_id, "block_id": block.block_id,
            "char_start": span.start, "char_end": span.end,
            "source_slice_sha256": sha(text[span.start:span.end].encode())}
    result = []
    regions = {r["id"]: r for r in document.parse_provenance["source_review"]["regions"]}
    for key, mapping in region_block_mapping(document).items():
        region = regions[key]
        page, = [p for p in document.parse_provenance["source_page_spans"] if p["page_index"] == region["page_index"]]
        start = text.find(region["raw_candidate"], page["char_start"], page["char_end"])
        assert start >= 0
        end = start + len(region["raw_candidate"])
        linked = [blocks[key] for key in mapping["block_ids"]]
        assert linked and all(b["char_start"] < end and start < b["char_end"] for b in linked)
        result.append({"region_id": key, "char_start": start, "char_end": end,
                       "mapping_status": mapping["mapping_status"], "linked_blocks": linked})
    return result


def partial_page_case(document, evidence):
    """A declared partial-layout fixture over the same immutable accepted source."""
    cut = (evidence["page_char_start"] + evidence["page_char_end"]) // 2
    chapters = []
    for chapter in document.chapters:
        sections = []
        for section in chapter.sections:
            if section.char_start < cut < section.char_end:
                left_text = section.content[:cut-section.char_start]
                left = replace(section, content=left_text, char_end=cut, task_units=[TaskUnit(
                    unit_id="partial-real-unit", title=None, container_title=None, content=left_text,
                    source_section_ids=[section.section_id], is_fallback_generated=False,
                    parent_section_id=section.section_id)])
                right = replace(section, section_id=section.section_id + "-pending",
                    section_index=section.section_index + 1, char_start=cut,
                    content=section.content[cut-section.char_start:], task_units=[])
                sections.extend((left, right))
            else:
                sections.append(section)
        chapters.append(replace(chapter, sections=sections))
    result = replace(document, chapters=chapters)
    assert canonical_text(result) == canonical_text(document)
    assert region_block_mapping(result)["real-page"]["mapping_status"] == "partial"
    return result


@contextmanager
def repository(backend):
    if backend == "file":
        with tempfile.TemporaryDirectory(prefix="srmap07-") as directory:
            yield StructuredDocumentArtifactRepository(base_dir=directory)
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from db.postgres_structured_document_store import PostgresStructuredDocumentStore
    from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
    dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    schema = "srmap07_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        scoped = make_conninfo(dsn, options=f"-c search_path={schema}")
        with psycopg.connect(scoped) as connection:
            for path in sorted((Path(__file__).resolve().parents[1] / "db/migrations").glob("*.sql")):
                connection.execute(path.read_text())
        yield PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(scoped, namespace=uuid4().hex))
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def mutate(service, action, **kwargs):
    current = service.read("book")
    return service.mutate("book", action=action, expected_revision=current["revision"],
                          source_token=current["source_token"], mutation_id=uuid4().hex, **kwargs)


def exercise(repo, document, *, manual, partial=False):
    repo.save_reparsed_document(document, "book")
    stored = repo.load_document("book")
    before = deepcopy(stored.to_dict())
    service = ReadingContentService(repo)
    state = service.read("book")
    service.read("book", include_cleanup_audit=True)
    listing = repo.list_documents()
    assert repo.load_document("book").to_dict() == before, "read_changed_persistence"
    assert listing[0].review_required == (state["disposition"] == "needs_review")
    assert canonical_text(stored) == canonical_text(document)
    assert stored.parse_provenance["source_review"] == document.parse_provenance["source_review"]
    assert stored.parse_provenance["source_page_spans"] == document.parse_provenance["source_page_spans"]
    result = {"read_pure": True, "active_regions": len(state["regions"]),
              "mapping_persisted": "source_review_mapping" in stored.parse_provenance}
    if partial:
        mapping = region_block_mapping(stored)["real-page"]
        assert mapping["mapping_status"] == "partial" and mapping["body_status"] == "linked"
        result.update(partial_mapping=True, correspondence=correspondence(stored))
    if not manual:
        return result
    assert result["mapping_persisted"], "explicit_save_did_not_persist_mapping"
    result["correspondence"] = correspondence(stored)
    region = next(r for r in state["regions"] if r["id"] == "real-subquote")
    linked_ids = set(region["block_ids"])
    block = next(b for b in state["blocks"] if b["block_id"] in linked_ids)
    target = dict(target_level="content_block", task_unit_id=block["task_unit_id"], content_block_id=block["block_id"])
    notes = NoteInteractionService(repo)
    token = notes.read("book", **target)["target_token"]
    saved = notes.save("book", **target, text="SRMAP acceptance note", expected_revision=0,
                       mutation_id=uuid4().hex, target_token=token)
    edited = mutate(service, "edit", block_id=block["block_id"], text="SRMAP reviewed text", discard_interactions=False)
    related = [r for r in edited["regions"] if block["block_id"] in r["block_ids"]]
    assert len(related) == 2 and all(r["manual_change"]["status"] == "modified" for r in related)
    assert notes.read("book", **target)["note"] == saved["note"]
    for block_id in linked_ids:
        deleted = mutate(service, "delete", block_id=block_id)
    region = next(r for r in deleted["regions"] if r["id"] == "real-subquote")
    assert region["related_deleted_count"] == len(linked_ids)
    assert region["manual_change"]["status"] == "deleted"
    assert service.read("book", region_id="real-subquote")["blocks"] == []
    assert ReadingContentService(repo).read("book")["regions"] == deleted["regions"]
    # Single-level Undo restores the latest deletion, never purged interaction data.
    mutate(service, "undo")
    restored = service.read("book", region_id="real-subquote")
    assert restored["blocks"]
    try:
        notes.save("book", **target, text="Delayed note", expected_revision=0,
                   mutation_id=uuid4().hex, target_token=saved["target_token"])
    except NoteConflict:
        pass
    else:
        raise AssertionError("late_note_write_accepted")
    ignored = mutate(service, "discard_region", region_id="real-subquote")
    assert not any(r["id"] == "real-subquote" for r in ignored["regions"])
    assert any(r["id"] == "real-page" for r in ignored["regions"])
    after = repo.load_document("book")
    assert canonical_text(after) == canonical_text(document)
    assert after.parse_provenance["source_review"] == document.parse_provenance["source_review"]
    result.update(manual_edit_keep_note=True, last_delete_retains_review=True,
                  shared_region_metadata=True, explicit_ignore_only=True,
                  undo_restores_text=True, late_note_rejected=True, canonical_unchanged=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("file", "postgres", "both"), default="both")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or CORPUS_ROOT / "srmap07" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if not report_path.resolve().is_relative_to(CORPUS_ROOT.resolve()) or report_path.exists():
        parser.error("report must be a new file under the private test-books corpus")
    cases, hashes = load_cases()
    report = {"status": "running", "natural": {}, "constructed_real_source": {}, "runs": [],
              "backup_sha256": {str(p.relative_to(CORPUS_ROOT)): h for p, h in hashes.items()},
              "limitations": [], "runtime_sha256": {name: sha((Path(__file__).resolve().parents[1] / name).read_bytes())
                for name in ("document_structure/source_review_blocks.py", "document_structure/reading_content.py")}}
    try:
        for name, source, document in cases:
            report["natural"][name] = natural_evidence(document)
            report["natural"][name]["after_explicit_mapping_write"] = natural_evidence(persist_region_mapping(document))
            if name != "7270c4d2c581f525":
                derived, evidence = real_page_case(document)
                report["constructed_real_source"][name] = evidence
            else:
                derived = None
            for backend in (("file", "postgres") if args.backend == "both" else (args.backend,)):
                with repository(backend) as repo:
                    natural = exercise(repo, replace(document, source_path=str(source)), manual=False)
                    manual = exercise(repo, replace(derived, source_path=str(source)), manual=True) if derived else None
                    partial = exercise(repo, replace(partial_page_case(derived, evidence), source_path=str(source)),
                                       manual=False, partial=True) if derived else None
                    report["runs"].append({"case": name, "backend": backend,
                        "natural": natural, "manual": manual, "constructed_partial_layout": partial})
        if not any(v["body_status_counts"].get("linked") for v in report["natural"].values()):
            report["limitations"].append("No naturally linked review region in the four immutable backups; positive cases construct review rows from accepted real-page text without canonical_span.")
        report["limitations"].append("No OCR rerun, live model, HTTP/browser or production deployment tested.")
        report["status"] = "passed_with_documented_corpus_gap" if report["limitations"] else "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        raise
    finally:
        report["backups_unchanged"] = all(sha(path.read_bytes()) == expected for path, expected in hashes.items())
        preserve(report_path, json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True).encode())
        assert report["backups_unchanged"], "immutable_backup_changed"
        print(json.dumps({"status": report["status"], "report": str(report_path),
                          "completed_runs": len(report["runs"]), "backups_unchanged": report["backups_unchanged"]}))


if __name__ == "__main__":
    main()
