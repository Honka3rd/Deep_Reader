"""Explicit provenance-only TXT migration; never invoked by read/preflight."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def backfill(coordinator, name):
    from doc_loaders.text_document_loader import TextDocumentLoader
    from document_preparation.accepted_source_snapshot import source_file_sha256
    root = Path(coordinator.document_preparation_pipeline.loader_factory.base_dir).resolve()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("invalid_document_name")
    path = (root / (name + ".txt")).resolve()
    if path.parent != root or (root / (name + ".pdf")).exists():
        raise ValueError("source_ambiguous")
    loader = TextDocumentLoader(root)
    with coordinator.enhancement_lease(name):
        result = loader.extract(name)
        if source_file_sha256(path) != result.identity.source_sha256:
            raise ValueError("source_changed")
        return coordinator.document_artifact_repository.backfill_txt_extraction(
            name, result=result, source_path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-name", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Explicit --apply required for provenance migration")
    import main
    print("provenance_changed=" + str(backfill(main.section_task_coordinator, args.doc_name)))
