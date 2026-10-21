"""Immutable private book fixtures, independent of Library lifetime."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os


CORPUS_ROOT = Path(__file__).resolve().parents[1] / "data" / "test-books"


def preserve(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("backup_conflict: existing fixture must not be overwritten") from None


def fixture_path(case_id):
    if len(case_id) != 16 or any(c not in "0123456789abcdef" for c in case_id):
        raise ValueError("invalid_case_id")
    return CORPUS_ROOT / case_id


def backup_book(coordinator, doc_name):
    state = coordinator.read_enhancement_state(doc_name)
    before = coordinator.document_artifact_repository.load_enhancement_snapshot(doc_name)
    case_id = hashlib.sha256(doc_name.encode()).hexdigest()[:16]
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    CORPUS_ROOT.chmod(0o700)
    corpus = fixture_path(case_id)
    raw_root = Path(coordinator.document_preparation_pipeline.loader_factory.base_dir)
    source_format = state.source.extraction_identity.source_format
    source_name = "source." + source_format
    source = raw_root / (doc_name + "." + source_format)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != state.source.extraction_identity.source_sha256:
        raise ValueError("backup_source_changed")
    payloads = {source_name: raw,
        "snapshot.json": json.dumps(asdict(state.source), ensure_ascii=False, sort_keys=True).encode(),
        "original-document.json": json.dumps(before.document.to_dict(), ensure_ascii=False, sort_keys=True).encode()}
    for name, data in payloads.items():
        preserve(corpus / name, data)
    manifest = {"schema_version": 2, "source_format": source_format, "doc_name": doc_name,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    preserve(corpus / "manifest.json", json.dumps(manifest, sort_keys=True).encode())
    if coordinator.document_artifact_repository.load_enhancement_snapshot(doc_name) != before:
        raise ValueError("backup_document_changed")
    return case_id


def load_book(case_id):
    from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot, resolve_accepted_source_snapshot
    from doc_loaders.extraction_identity import ExtractionIdentity, normalize_pdf_identity
    from document_structure.structured_document import StructuredDocument
    corpus = fixture_path(case_id)
    manifest = json.loads((corpus / "manifest.json").read_text())
    source_format = "pdf" if manifest.get("schema_version") == 1 else manifest.get("source_format")
    if source_format not in ("pdf", "txt"):
        raise ValueError("invalid_backup_format")
    source_name = "source." + source_format
    names = {source_name, "snapshot.json", "original-document.json"}
    if manifest.get("schema_version") not in (1, 2) or set(manifest.get("files", {})) != names:
        raise ValueError("invalid_backup_manifest")
    data = {name: (corpus / name).read_bytes() for name in names}
    if any(hashlib.sha256(data[name]).hexdigest() != manifest["files"][name] for name in names):
        raise ValueError("backup_hash_mismatch")
    values = json.loads(data["snapshot.json"])
    identity = values["extraction_identity"]
    values["extraction_identity"] = (ExtractionIdentity(**identity) if "source_format" in identity
                                     else normalize_pdf_identity(identity))
    snapshot = AcceptedSourceSnapshot(**values)
    if (hashlib.sha256(data[source_name]).hexdigest() != snapshot.extraction_identity.source_sha256
            or hashlib.sha256(snapshot.canonical_text.encode()).hexdigest() != snapshot.text_sha256
            or len(snapshot.canonical_text) != snapshot.text_length):
        raise ValueError("backup_identity_mismatch")
    document = StructuredDocument.from_dict(json.loads(data["original-document.json"]))
    resolved = resolve_accepted_source_snapshot(document=document, namespace=snapshot.namespace,
        expected_document_id=snapshot.document_id, structure_version=snapshot.structure_version,
        source_path=corpus / source_name, accepted=True)
    if resolved != snapshot:
        raise ValueError("backup_structure_mismatch")
    return corpus, snapshot, document
