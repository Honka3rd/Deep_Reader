"""OCR-KEEP-06: actual-engine, backup-only preservation acceptance.

Run from any directory with Python 3.11+: python accept_ocr_preservation.py.
Only the Docker worker imports application code. No DB, API, profile or model
service is constructed. Reports are new private directories, never backup edits.
"""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from uuid import uuid4


BACKEND = Path(__file__).resolve().parents[1]
CORPUS = BACKEND / "data/test-books"
REPORT_ROOT = CORPUS / "ocr-keep-20261025"
DARK_HASH = "0fa63abf7e77b1681813727cd6ec4987bfc234750122edaf908d8016ebcb044d"
SAMPLE_HASH = "05650a15d6a0428d543e4ffee273e5423ecdbcecf14c7ae0f4a02b16d13574de"
PHRASE = "\u88d5\u4e4b\u5e76\u4e0d\u5341\u5206\u5728\u4e4e\u8fd9\u4e9b"
PARAGRAPH_ANCHORS = (
    ("middle_1", ("\u4ed6\u624d\u4e0d\u7ba1",)),
    ("middle_2", ("\u6bcf\u6708\u80fd\u628a",)),
    ("ending", ("\u8aaa\u4e09\u9053\u56db", "\u8bf4\u4e09\u9053\u56db")),
)


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text):
    return re.sub(r"\s", "", text)


def private_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)


def backup_paths():
    dark = CORPUS / "7270c4d2c581f525"
    sample = CORPUS / "ocr-review-20261021"
    manifest = json.loads((sample / "manifest.json").read_text())
    sample_pdf = (sample / manifest["source"]).resolve()
    require(sample_pdf.is_relative_to(sample.resolve()), "sample_source_outside_backup")
    require(manifest["sha256"] == SAMPLE_HASH, "sample_manifest_identity_changed")
    paths = [dark / name for name in (
        "source.pdf", "manifest.json", "snapshot.json", "original-document.json", "profile.json")]
    paths.extend([sample / "manifest.json", sample_pdf])
    paths.extend(p for p in (sample / "acceptance.json", sample / "reading-review-acceptance.json") if p.exists())
    return dark / "source.pdf", sample_pdf, manifest, paths


def file_hashes(paths, root):
    return {str(path.relative_to(root)): sha(path) for path in paths}


def runtime_hashes():
    paths = [p for folder in ("doc_loaders", "document_structure", "section_tasks", "shared", "language")
             for p in (BACKEND / folder).rglob("*.py")]
    paths.append(Path(__file__).resolve())
    return file_hashes(sorted(paths), BACKEND)


def materialize(document):
    from section_tasks.task_unit_resolver import TaskUnitResolver
    from section_tasks.task_unit_id_normalizer import TaskUnitIdNormalizer

    units = TaskUnitResolver(split_mode="progressive").resolve_for_task_layout(document=document)
    require(bool(units), "no_task_units")
    require(all(len(u.source_section_ids) == 1 for u in units), "cross_section_task_unit")
    document = replace(document, chapters=[replace(chapter, sections=[replace(section,
        task_units=[u for u in units if u.source_section_ids == [section.section_id]])
        for section in chapter.sections]) for chapter in document.chapters])
    return TaskUnitIdNormalizer().normalize_document_task_unit_ids(document=document)


def paragraph_evidence(layers, phrase, anchors):
    from document_structure.text_normalization import normalize_ocr_text

    evidence = {"layers": {}, "complete": False}
    groups = (("opening", (phrase,)), *anchors)
    for name, body in layers.items():
        text = normalized(body)
        cursor = 0
        found = []
        for label, variants in groups:
            matches = [(text.find(normalized(value), cursor), value) for value in variants]
            matches = [(start, value) for start, value in matches if start >= 0]
            if not matches:
                found.append({"anchor": label, "found": False})
                break
            start, variant = min(matches)
            cursor = start + len(normalized(variant))
            found.append({"anchor": label, "found": True, "variant": variant,
                          "normalized_start": start, "normalized_end": cursor})
        evidence["layers"][name] = {"ordered_anchors": found,
            "all_anchors_found": len(found) == len(groups) and all(hit["found"] for hit in found)}
    if not all(layer["all_anchors_found"] for layer in evidence["layers"].values()):
        return evidence
    canonical = layers["canonical"]
    offsets = [i for i, character in enumerate(canonical) if not character.isspace()]
    hits = evidence["layers"]["canonical"]["ordered_anchors"]
    first = offsets[hits[0]["normalized_start"]]
    last = offsets[hits[-1]["normalized_end"] - 1] + 1
    separators = list(re.finditer(r"\n\s*\n", canonical))
    start = max((match.end() for match in separators if match.end() <= first), default=0)
    end = min((match.start() for match in separators if match.start() >= first), default=len(canonical))
    evidence["anchors_in_same_paragraph"] = last <= end
    paragraph = canonical[start:end]
    evidence["canonical_span"] = {"char_start": start, "char_end": end,
                                   "text_sha256": text_sha(canonical)}
    evidence["paragraph_sha256"] = text_sha(paragraph)
    evidence["paragraph_characters"] = len(paragraph)
    for name, body in layers.items():
        expected = normalize_ocr_text(paragraph) if name == "reading_blocks" else paragraph
        evidence["layers"][name]["whole_paragraph_present"] = normalized(expected) in normalized(body)
    evidence["complete"] = evidence["anchors_in_same_paragraph"] and all(
        layer["whole_paragraph_present"] for layer in evidence["layers"].values())
    return evidence


def verify_body(document, artifact, phrase, observations=None, paragraph_anchors=()):
    from document_structure.ocr_cleanup_overlay import canonical_text, block_range, unit_ranges
    from document_structure.reading_content import original_blocks
    from document_structure.source_review_blocks import region_block_mapping, active_review_regions
    from document_structure.text_normalization import normalize_ocr_text

    text = artifact.result.canonical_text
    digest = text_sha(text)
    require(canonical_text(document) == text, "common_canonical_text_changed")
    sections = [s for c in document.chapters for s in c.sections]
    require(normalized("".join(s.content for s in sections)) == normalized(text), "common_lost_text")
    for section in sections:
        require(text[section.char_start:section.char_end] == section.content, "section_span_mismatch")
    require(normalized("".join(u.content for s in sections for u in s.task_units)) == normalized(text),
            "task_units_lost_text")
    blocks = {}
    block_sources = []
    display_changes = []
    cursor = 0
    for chapter, section, unit, block in original_blocks(document):
        span = block_range(unit, block, unit_ranges(section)[unit.unit_id])
        source = text[span.start:span.end]
        require(source == block.content or (
            block.metadata.get("display_normalization") == "ocr_whitespace_v1"
            and normalize_ocr_text(source) == block.content), "block_source_mismatch")
        require(cursor <= span.start < span.end <= len(text), "block_source_ranges_overlap_or_reorder")
        require(not text[cursor:span.start].strip(), "blocks_omit_source_before_range")
        cursor = span.end
        require(block.block_id not in blocks, "duplicate_block_id")
        block_sources.append(source)
        if source != block.content:
            left, right = normalized(source), normalized(block.content)
            display_changes.append({"block_id": block.block_id,
                "policy": block.metadata.get("display_normalization"),
                "source_sha256": text_sha(source), "display_sha256": text_sha(block.content),
                "source_nonspace_length": len(left), "display_nonspace_length": len(right),
                "nonspace_changes": [{"operation": tag, "source_offset": i, "display_offset": k,
                    "source_codepoints": [f"U+{ord(c):04X}" for c in left[i:j]],
                    "display_codepoints": [f"U+{ord(c):04X}" for c in right[k:l]]}
                    for tag, i, j, k, l in difflib.SequenceMatcher(None, left, right).get_opcodes()
                    if tag != "equal"]})
        blocks[block.block_id] = {"task_unit_id": unit.unit_id, "chapter_id": chapter.chapter_id,
            "section_id": section.section_id, "char_start": span.start, "char_end": span.end,
            "source_sha256": text_sha(source), "content": block.content}
    require(bool(blocks), "no_reading_blocks")
    require(not text[cursor:].strip(), "blocks_omit_source_tail")
    require(normalized("".join(block_sources)) == normalized(text), "blocks_lost_source_text")
    layers = {"canonical": text, "common_sections": "".join(s.content for s in sections),
              "task_units": "".join(u.content for s in sections for u in s.task_units),
              "reading_blocks": "".join(b["content"] for b in blocks.values())}
    phrase_checks = {key: normalized(phrase) in normalized(body) for key, body in layers.items()}
    paragraph = paragraph_evidence(layers, phrase, paragraph_anchors) if paragraph_anchors else None
    coverage = [{"block_id": key, "char_start": b["char_start"], "char_end": b["char_end"],
                 "source_sha256": b["source_sha256"]} for key, b in blocks.items()]
    if observations is not None:
        observations.update(phrase_checks=phrase_checks, display_normalizations=display_changes,
                            block_source_coverage=coverage, complete_nonspace_source_coverage=True,
                            paragraph=paragraph)
    pages = {p.page_index: p for p in artifact.page_boundaries}
    for page in pages.values():
        require(text[page.char_start:page.char_end] == page.text, "page_span_mismatch")
        require(page.source_sha256 == artifact.result.identity.source_sha256, "page_identity_mismatch")
    before = document.to_dict()
    mappings = region_block_mapping(document)
    require(document.to_dict() == before, "mapping_read_mutated_document")
    regions = document.parse_provenance.get("source_review", {}).get("regions", [])
    active_ids = {r["id"] for r in active_review_regions(document.parse_provenance)}
    rows = []
    for region in regions:
        require(region["id"] in active_ids, "natural_warning_hidden")
        candidate = region.get("raw_candidate", "")
        mapping = mappings[region["id"]]
        row = {"region_id": region["id"], "page_index": region["page_index"],
               "reason": region["reason"], "candidate_length": len(candidate),
               "candidate_sha256": text_sha(candidate), **mapping}
        if candidate:
            require(region.get("body_inclusion") == "included", "nonempty_region_not_included")
            span = region.get("canonical_span", {})
            start, end = span.get("char_start"), span.get("char_end")
            page = pages[region["page_index"]]
            require(type(start) is int and type(end) is int, "missing_exact_review_span")
            require(page.char_start <= start < end <= page.char_end, "review_outside_source_page")
            require(span.get("text_sha256") == digest and text[start:end] == candidate, "review_span_mismatch")
            expected_ids = {key for key, b in blocks.items() if b["char_start"] < end and start < b["char_end"]}
            require(set(mapping["block_ids"]) == expected_ids and bool(expected_ids), "review_wrong_body_blocks")
            require(mapping["body_status"] == "linked" and mapping["mapping_status"] == "complete",
                    "review_mapping_not_complete")
            row.update(canonical_span=span, linked_blocks=[
                {k: v for k, v in blocks[key].items() if k != "content"} for key in mapping["block_ids"]])
        else:
            require(not region.get("canonical_span") and not mapping["block_ids"], "empty_region_invented_body")
        rows.append(row)
    result = {"canonical_length": len(text), "canonical_sha256": digest,
            "page_characters": [len(p.text) for p in artifact.page_boundaries],
            "phrase_checks": phrase_checks, "section_count": len(sections),
            "task_unit_count": sum(len(s.task_units) for s in sections), "block_count": len(blocks),
            "natural_review_count": len(rows), "review_regions": rows,
            "natural_positive_mapping": any(row["candidate_length"] for row in rows),
            "mapping_read_pure": True, "block_source_coverage": coverage,
            "display_normalizations": display_changes, "complete_nonspace_source_coverage": True,
            "paragraph": paragraph}
    if observations is not None:
        observations.update(result)
    require(all(phrase_checks.values()), "expected_phrase_missing:" + str(phrase_checks))
    if paragraph is not None:
        require(paragraph["complete"], "incident_paragraph_not_preserved:" + str(paragraph))
    return result


def extract_case(source, mode, phrase, output, name, original_indices, manifest=None):
    from doc_loaders.pdf_document_loader import PdfDocumentLoader
    from document_structure.structured_document_builder import StructuredDocumentBuilder
    from language.language_code import LanguageCode

    started = time.monotonic()
    loader = PdfDocumentLoader(source.parent, ocr_enabled=True, ocr_page_limit=3)
    session = loader.open_source(source.stem, mode=mode)
    artifact = session.extract()
    session.verify_source()
    require(len(artifact.page_boundaries) == len(original_indices), "wrong_extracted_page_count")
    metadata = dict(artifact.storage_metadata)
    metadata.pop("pdf_extraction", None)
    metadata["source_page_spans"] = metadata.pop("pdf_page_spans")
    metadata["extraction"] = artifact.result.identity.record(artifact.result.canonical_text)
    quality = artifact.ocr_run["provenance"]["page_quality"]
    metadata["source_page_quality"] = quality
    # Keep extraction-owned spans/identity verbatim, as in the existing corpus harness.
    document = StructuredDocumentBuilder().build(document_id=name, title=name,
        raw_text=artifact.result.canonical_text, language=LanguageCode.ZH,
        source_path=str(source), parser_mode="common", page_boundaries=list(artifact.page_boundaries))
    require(not document.structure_error_code, "common_parse_failed:" + str(document.structure_error_code))
    document = replace(document, parse_provenance={**document.parse_provenance, **metadata})
    document = materialize(document)
    private_json(output / (name + "-document.json"), document.to_dict())
    private_json(output / (name + "-ocr.json"), artifact.ocr_run)
    observations = {
        "mode": mode, "source_sha256": sha(source), "original_page_indices": original_indices,
        "canonical_length": len(artifact.result.canonical_text),
        "canonical_sha256": text_sha(artifact.result.canonical_text),
        "phrase_in_canonical": normalized(phrase) in normalized(artifact.result.canonical_text),
        "page_characters": [len(p.text) for p in artifact.page_boundaries],
        "page_quality": quality,
    }
    try:
        result = verify_body(document, artifact, phrase, observations,
                             paragraph_anchors=PARAGRAPH_ANCHORS if name.startswith("dark-page80-") else ())
    finally:
        private_json(output / (name + "-observations.json"), observations)
    result.update(name=name, mode=mode, source_sha256=sha(source),
        original_page_indices=original_indices,
        page_index_map=[{"extracted_index": i, "original_index": original, "physical_page": original + 1}
                        for i, original in enumerate(original_indices)],
        extraction=metadata["extraction"], seconds=round(time.monotonic() - started, 3),
        quality_selected_passed=[p.get("selected_passed") for p in quality])
    if manifest:
        require(len(artifact.page_boundaries) == manifest["page_count"], "sample2_incomplete")
        require([p.page_index for p in artifact.page_boundaries if not p.text] == manifest["blank_page_indexes"],
                "sample2_blank_page_changed")
        require(quality[1].get("blank_render_confirmed") is True, "sample2_blank_unconfirmed")
        margins = [r for r in quality[2]["regions"] if r.get("region_role") == "margin_label"]
        require(len(margins) == 1, "sample2_margin_missing")
        margin = margins[0]
        page = artifact.page_boundaries[2]
        require(page.text[margin["char_start"]:margin["char_end"]] == manifest["page_3_margin"],
                "sample2_margin_changed")
        require([c["role"] for c in margin["components"]] == ["title", "page_digit", "page_digit", "page_digit"],
                "sample2_margin_components_changed")
        result.update(sample2_all_pages=True, margin_exact=True, blank_render_confirmed=True)
    return result


def legacy_gap():
    from document_structure.structured_document import StructuredDocument
    from document_structure.source_review_blocks import active_review_regions, region_block_mapping

    document = StructuredDocument.from_dict(json.loads(
        (CORPUS / "7270c4d2c581f525/original-document.json").read_text()))
    before = document.to_dict()
    provenance = document.parse_provenance
    regions = [r for r in provenance["source_review"]["regions"] if r["page_index"] == 79]
    require(bool(regions), "legacy_incident_evidence_missing")
    page, = [p for p in provenance["source_page_spans"] if p["page_index"] == 79]
    require(page["char_start"] == page["char_end"], "legacy_gap_no_longer_reproduced")
    active = {r["id"] for r in active_review_regions(provenance)}
    mappings = region_block_mapping(document)
    for region in regions:
        require(region["id"] in active, "legacy_gap_hidden_from_review")
        require(not mappings[region["id"]]["block_ids"], "legacy_gap_invented_body")
        require(mappings[region["id"]]["body_status"] != "excluded", "legacy_gap_silently_excluded")
    require(document.to_dict() == before, "legacy_read_mutated_document")
    return {"page_index": 79, "physical_page": 80, "original_span": page,
            "active_region_ids": [r["id"] for r in regions],
            "mappings": {r["id"]: mappings[r["id"]] for r in regions}, "read_pure": True}


def worker(output):
    from pypdf import PdfReader, PdfWriter

    require(Path("/.dockerenv").exists() and output.resolve() == Path("/output"), "worker_requires_isolated_container")
    dark, sample, manifest, paths = backup_paths()
    processes = []

    def observe_process(event, args):
        if event == "subprocess.Popen":
            command = args[1]
            if isinstance(command, (tuple, list)) and command:
                program = Path(command[0]).name
                if program in {"tesseract", "pdftoppm"}:
                    processes.append({"program": program,
                        "recognition": program == "tesseract" and "stdout" in command,
                        "render": program == "pdftoppm" and "-v" not in command,
                        "psm": command[command.index("--psm") + 1] if "--psm" in command else None})

    sys.addaudithook(observe_process)
    report = {"status": "running", "cases": [], "errors": [], "processes": {}, "limitations": [
        "Only incident physical page 80 and all three Sample2 pages are OCRed; no 261-page OCR.",
        "Preservation and representative phrases are checked, not full transcription accuracy.",
        "Known OCR numeral errors are not corrected or certified by paragraph preservation.",
        "No HTTP/browser, production deployment, database, profile or LLM acceptance in this harness."],
        "platform": platform.platform(), "python": platform.python_version(),
        "backup_hashes_before": file_hashes(paths, CORPUS), "runtime_hashes_before": runtime_hashes()}
    try:
        require(sha(dark) == DARK_HASH and sha(sample) == SAMPLE_HASH, "backup_source_hash_mismatch")
        for command in (["tesseract", "--version"], ["pdftoppm", "-v"]):
            version = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
            report[command[0]] = (version.stdout + version.stderr).splitlines()[0]
        with tempfile.TemporaryDirectory(prefix="ocr-keep-") as temporary:
            excerpt = Path(temporary) / "dark-page80.pdf"
            reader = PdfReader(dark)
            require(len(reader.pages) == 261, "wrong_incident_book_page_count")
            writer = PdfWriter()
            writer.add_page(reader.pages[79])
            with excerpt.open("xb") as stream:
                writer.write(stream)
            report["excerpt"] = {"original_source_sha256": DARK_HASH, "source_sha256": sha(excerpt),
                                 "original_page_indices": [79], "page_count": 1}
            for name, source, mode, phrase, indices, sample_manifest in (
                ("dark-page80-ocr", excerpt, "ocr", PHRASE, [79], None),
                ("dark-page80-auto", excerpt, "auto", PHRASE, [79], None),
                ("sample2-ocr", sample, "ocr", manifest["page_3_body_phrase"], [0, 1, 2], manifest),
            ):
                print(json.dumps({"case": name, "status": "started"}), flush=True)
                process_start = len(processes)
                try:
                    result = extract_case(source, mode, phrase, output, name, indices, sample_manifest)
                    require(any(p["recognition"] for p in processes[process_start:]), "no_actual_ocr_process")
                    require(any(p["render"] for p in processes[process_start:]), "no_actual_render_process")
                    report["cases"].append(result)
                    print(json.dumps({"case": name, "status": "passed",
                        "page_characters": result["page_characters"],
                        "natural_review_count": result["natural_review_count"]}), flush=True)
                except Exception as error:
                    failure = {"case": name, "type": type(error).__name__,
                               "message": str(error), "traceback": traceback.format_exc()}
                    report["errors"].append(failure)
                    private_json(output / (name + "-failure.json"), failure)
                    print(json.dumps({"case": name, "status": "failed", "type": type(error).__name__}), flush=True)
                finally:
                    report["processes"][name] = {
                        "attempts": processes[process_start:],
                        "counts": dict(Counter(p["program"] for p in processes[process_start:]))}
            try:
                report["legacy_gap"] = legacy_gap()
            except Exception as error:
                report["errors"].append({"case": "legacy_gap", "type": type(error).__name__,
                                         "message": str(error), "traceback": traceback.format_exc()})
        for case in report["cases"]:
            if not case["natural_positive_mapping"]:
                report["limitations"].append(case["name"] + ": no naturally retained quality-warning region; positive review mapping not exercised for this case.")
        report["status"] = "failed" if report["errors"] else "passed_with_scoped_limitations"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        report["backup_hashes_after"] = file_hashes(paths, CORPUS)
        report["backups_unchanged"] = report["backup_hashes_before"] == report["backup_hashes_after"]
        report["runtime_hashes_after"] = runtime_hashes()
        report["runtime_unchanged"] = report["runtime_hashes_before"] == report["runtime_hashes_after"]
        report["runtime_changed_files"] = [key for key in
            sorted(report["runtime_hashes_before"].keys() | report["runtime_hashes_after"].keys())
            if report["runtime_hashes_before"].get(key) != report["runtime_hashes_after"].get(key)]
        if not report["backups_unchanged"] or not report["runtime_unchanged"]:
            report["status"] = "failed"
        private_json(output / "report.json", report)
    return 1 if report["status"] == "failed" else 0


def docker_command(image, temporary, name):
    return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,size=1g",
        "--mount", f"type=bind,src={BACKEND.parent},dst=/repo,readonly",
        "--mount", f"type=bind,src={BACKEND},dst=/app/Deep_Reflective_Reader,readonly",
        "--mount", f"type=bind,src={temporary},dst=/output",
        "--workdir", "/app/Deep_Reflective_Reader", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONPATH=/app/Deep_Reflective_Reader", "--env", "OMP_THREAD_LIMIT=1",
        "--entrypoint", "python", image, "-B", "-m", "scripts.accept_ocr_preservation", "--worker"]


def host(image):
    os.umask(0o077)
    _, _, _, paths = backup_paths()
    before = file_hashes(paths, CORPUS)
    require(sha(CORPUS / "7270c4d2c581f525/source.pdf") == DARK_HASH, "incident_backup_hash_mismatch")
    image_id = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                              capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    name = "ocr-keep-" + uuid4().hex[:12]
    destination = REPORT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with tempfile.TemporaryDirectory(prefix="ocr-keep-output-") as temporary:
        command = docker_command(image_id, temporary, name)
        status = 1
        error = None
        try:
            with (Path(temporary) / "worker.log").open("x") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as caught:
            error = type(caught).__name__
        finally:
            # A killed Docker client must not leave its worker running.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            after = file_hashes(paths, CORPUS)
            receipt = {"image": image, "image_id": image_id, "command": command, "exit_code": status,
                "runner_error": error, "backup_hashes_before": before, "backup_hashes_after": after,
                "backups_unchanged": before == after, "report_directory": str(destination)}
            private_json(Path(temporary) / "isolation.json", receipt)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(temporary, destination)
        report_file = destination / "report.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {"status": "worker_failed"}
        print(json.dumps({"status": report["status"], "report": str(report_file),
            "backups_unchanged": before == after, "completed_cases": len(report.get("cases", []))}))
        return 0 if status == 0 and before == after and report.get("status") == "passed_with_scoped_limitations" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="deep_reader-api")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(Path("/output")) if args.worker else host(args.image)


if __name__ == "__main__":
    raise SystemExit(main())
