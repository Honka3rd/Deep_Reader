#!/usr/bin/env python3
"""Regression tests for manual-structure commit source-evidence gates."""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.section_task_coordinator import (  # noqa: E402
    ManualStructureAnchorDTO,
    ManualStructureEntryDTO,
    ManualStructurePlanDTO,
    SectionTaskCoordinator,
)
from document_preparation.document_preparation_pipeline import (  # noqa: E402
    ManualStructureSourceEvidence,
)
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary  # noqa: E402


class _FakeLoader:
    def __init__(
        self,
        *,
        raw_text: str | None = None,
        error: Exception | None = None,
        page_boundaries: list[PdfPageTextBoundary] | None = None,
    ) -> None:
        self.raw_text = raw_text
        self.error = error
        self.page_boundaries = page_boundaries or []
        self.load_calls: list[str] = []

    def load(self, doc_name: str) -> str:
        self.load_calls.append(doc_name)
        if self.error is not None:
            raise self.error
        return self.raw_text or ""


class _FakeLoaderFactory:
    def __init__(self, loader: _FakeLoader) -> None:
        self.loader = loader
        self.get_calls: list[str] = []

    def get(self, doc_name: str) -> _FakeLoader:
        self.get_calls.append(doc_name)
        return self.loader


class _FakePreparationPipeline:
    def __init__(self, loader: _FakeLoader) -> None:
        self.loader = loader
        self.loader_factory = _FakeLoaderFactory(loader)
        self.source_evidence_calls: list[str] = []

    def load_manual_structure_source_evidence(
        self,
        doc_name: str,
    ) -> ManualStructureSourceEvidence:
        self.source_evidence_calls.append(doc_name)
        raw_text = self.loader_factory.get(doc_name).load(doc_name)
        if raw_text is None or not raw_text.strip():
            raise ValueError(f"manual_structure source is empty for doc_name='{doc_name}'")
        return ManualStructureSourceEvidence(
            doc_name=doc_name,
            source_identity=doc_name,
            raw_text=raw_text,
            source_hash=_hash(raw_text),
            language="en",
            page_boundaries=list(self.loader.page_boundaries),
            errors=[],
        )


class _FakeRepository:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.save_calls: list[tuple[object, str | None]] = []
        self.save_document_calls: list[tuple[object, str | None]] = []
        self.save_reparsed_document_calls: list[tuple[object, str | None]] = []

    def save_document(self, document, doc_name=None):  # noqa: ANN001
        self.save_document_calls.append((document, doc_name))
        self.save_calls.append((document, doc_name))
        if self.error is not None:
            raise self.error

    def save_reparsed_document(self, document, doc_name=None):  # noqa: ANN001
        self.save_reparsed_document_calls.append((document, doc_name))
        self.save_calls.append((document, doc_name))
        if self.error is not None:
            raise self.error


def _coordinator(
    loader: _FakeLoader,
    repository: _FakeRepository | None = None,
) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(loader),  # type: ignore[arg-type]
        document_artifact_repository=repository or _FakeRepository(),  # type: ignore[arg-type]
        document_profile_store=None,  # type: ignore[arg-type]
        chapter_summary_service=None,  # type: ignore[arg-type]
        chapter_quiz_service=None,  # type: ignore[arg-type]
        task_unit_resolver=None,  # type: ignore[arg-type]
        enhanced_parse_trigger_evaluator=None,  # type: ignore[arg-type]
    )


def _plan(*, source_hash: str | None = None) -> ManualStructurePlanDTO:
    return ManualStructurePlanDTO(
        source_hash=source_hash,
        entries=[
            ManualStructureEntryDTO(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=100,
                ),
            )
        ],
    )


def _out_of_bounds_plan() -> ManualStructurePlanDTO:
    return ManualStructurePlanDTO(
        entries=[
            ManualStructureEntryDTO(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=100,
                ),
            )
        ],
    )


def _overlap_plan() -> ManualStructurePlanDTO:
    return ManualStructurePlanDTO(
        entries=[
            ManualStructureEntryDTO(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="char_range",
                    char_start=0,
                    char_end=100,
                ),
            ),
            ManualStructureEntryDTO(
                title="Chapter Two",
                level=1,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="char_range",
                    char_start=90,
                    char_end=150,
                ),
            ),
        ],
    )


def _page_range_plan(*, source_hash: str | None = None) -> ManualStructurePlanDTO:
    return ManualStructurePlanDTO(
        source_hash=source_hash,
        entries=[
            ManualStructureEntryDTO(
                title="Chapter One",
                level=1,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="page_range",
                    page_start_index=0,
                    page_end_index=1,
                ),
            ),
            ManualStructureEntryDTO(
                title="Section One",
                level=2,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="page_range",
                    page_start_index=0,
                    page_end_index=0,
                ),
            ),
            ManualStructureEntryDTO(
                title="Section Two",
                level=2,
                anchor=ManualStructureAnchorDTO(
                    anchor_type="page_range",
                    page_start_index=1,
                    page_end_index=1,
                ),
            ),
        ],
    )


def _hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_invalid_projection_does_not_load_source() -> None:
    loader = _FakeLoader(raw_text="current source")
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.txt",
        manual_structure=_overlap_plan(),
    )

    _assert(result.status_code == 422, f"unexpected status: {result.status_code}")
    _assert("overlapping_range" in (result.error or ""), f"unexpected error: {result.error}")
    _assert(loader.load_calls == [], "invalid projection should not load source evidence")
    _assert(repository.save_calls == [], "invalid projection should not save hierarchy")


def test_missing_source_returns_404_before_commit() -> None:
    loader = _FakeLoader(error=FileNotFoundError("data/raw/Missing.txt not found"))
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Missing.txt",
        manual_structure=_plan(),
    )

    _assert(result.status_code == 404, f"unexpected status: {result.status_code}")
    _assert(result.structured_document_path is None, "missing source must not persist hierarchy")
    _assert(result.section_count is None, "missing source must not produce section count")
    _assert("source not found" in (result.error or ""), f"unexpected error: {result.error}")
    _assert(repository.save_calls == [], "missing source should not save hierarchy")


def test_source_hash_mismatch_returns_409_before_commit() -> None:
    loader = _FakeLoader(raw_text="current source")
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.txt",
        manual_structure=_plan(source_hash=_hash("stale source")),
    )

    _assert(result.status_code == 409, f"unexpected status: {result.status_code}")
    _assert(result.structured_document_path is None, "stale source must not persist hierarchy")
    _assert(result.section_count is None, "stale source must not produce section count")
    _assert("stale" in (result.error or ""), f"unexpected error: {result.error}")
    _assert(repository.save_calls == [], "stale source should not save hierarchy")


def test_draft_build_rejects_raw_text_out_of_bounds_before_commit() -> None:
    loader = _FakeLoader(raw_text="short")
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.txt",
        manual_structure=_out_of_bounds_plan(),
    )

    _assert(result.status_code == 422, f"unexpected status: {result.status_code}")
    _assert(result.structured_document_path is None, "invalid draft must not persist hierarchy")
    _assert(result.section_count is None, "invalid draft must not produce section count")
    _assert("out_of_range_anchor" in (result.error or ""), f"unexpected error: {result.error}")
    _assert(repository.save_calls == [], "invalid draft should not save hierarchy")


def test_matching_source_hash_commits_manual_structure_document() -> None:
    raw_text = "Chapter One\n" + ("current source " * 10)
    loader = _FakeLoader(raw_text=raw_text)
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.txt",
        manual_structure=_plan(source_hash=_hash(raw_text)),
    )

    _assert(result.success, f"unexpected failure: {result.error}")
    _assert(result.status_code == 200, f"unexpected status: {result.status_code}")
    _assert(result.error is None, f"unexpected error: {result.error}")
    _assert(result.section_count == 1, f"unexpected section count: {result.section_count}")
    _assert(loader.load_calls == ["Book.txt"], "valid commit path should load source evidence")
    _assert(len(repository.save_calls) == 1, "valid commit path should save exactly once")
    _assert(
        repository.save_document_calls == [],
        "manual commit must not use generic artifact save boundary",
    )
    _assert(
        len(repository.save_reparsed_document_calls) == 1,
        "manual commit should use reparse replacement save boundary",
    )
    saved_document, saved_doc_name = repository.save_calls[0]
    _assert(saved_doc_name == "Book.txt", "manual commit should save by normalized doc name")
    _assert(saved_document.sections == [], "manual commit must not persist root sections")
    _assert(saved_document.structure_nodes == [], "manual commit must not persist structure_nodes")
    _assert(len(saved_document.chapters) == 1, "manual commit should save draft chapters")
    _assert(saved_document.language == "en", "manual commit should preserve evidence language")
    _assert(
        saved_document.parse_provenance["manual_structure"]["source_hash"]
        == _hash(raw_text),
        "manual provenance should record actual source hash",
    )


def test_page_range_commit_uses_preparation_page_boundaries() -> None:
    page_1 = "Chapter One\nSection One body."
    page_2 = "Section Two\nSection Two body."
    raw_text = f"{page_1}\n\n{page_2}"
    page_2_start = len(page_1) + 2
    loader = _FakeLoader(
        raw_text=raw_text,
        page_boundaries=[
            PdfPageTextBoundary(
                page_index=0,
                char_start=0,
                char_end=len(page_1),
                text=page_1,
                page_label="1",
            ),
            PdfPageTextBoundary(
                page_index=1,
                char_start=page_2_start,
                char_end=len(raw_text),
                text=page_2,
                page_label="2",
            ),
        ],
    )
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.pdf",
        manual_structure=_page_range_plan(source_hash=_hash(raw_text)),
    )

    _assert(result.success, f"unexpected failure: {result.error}")
    _assert(result.status_code == 200, f"unexpected status: {result.status_code}")
    _assert(len(repository.save_calls) == 1, "page_range commit should save once")
    _assert(
        len(repository.save_reparsed_document_calls) == 1,
        "page_range commit should use reparse replacement save boundary",
    )
    saved_document, _ = repository.save_calls[0]
    _assert(saved_document.sections == [], "page_range commit must not persist root sections")
    _assert(saved_document.structure_nodes == [], "page_range commit must not persist structure_nodes")
    _assert(
        [section.content for section in saved_document.chapters[0].sections]
        == [page_1, page_2],
        "page_range commit should slice section content from page boundaries",
    )
    _assert(
        saved_document.parse_provenance["manual_structure"]["anchor_type"]
        == "page_range",
        "page_range commit provenance should preserve anchor type",
    )


def test_page_range_commit_rejects_ambiguous_page_boundaries_before_save() -> None:
    page_1 = "Chapter One\nSection One body."
    page_2 = "Section Two\nSection Two body."
    raw_text = f"{page_1}\n\n{page_2}"
    page_2_start = len(page_1) + 2
    loader = _FakeLoader(
        raw_text=raw_text,
        page_boundaries=[
            PdfPageTextBoundary(
                page_index=0,
                char_start=0,
                char_end=len(page_1),
                text=page_1,
                page_label="1",
            ),
            PdfPageTextBoundary(
                page_index=0,
                char_start=page_2_start,
                char_end=len(raw_text),
                text=page_2,
                page_label="duplicate-1",
            ),
        ],
    )
    repository = _FakeRepository()
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.pdf",
        manual_structure=_page_range_plan(source_hash=_hash(raw_text)),
    )

    _assert(not result.success, "ambiguous page evidence must not report success")
    _assert(result.status_code == 422, f"unexpected status: {result.status_code}")
    _assert(result.structured_document_path is None, "invalid page evidence must not report a path")
    _assert(result.section_count is None, "invalid page evidence must not report committed sections")
    _assert(
        "ambiguous_page_boundary" in (result.error or ""),
        f"unexpected error: {result.error}",
    )
    _assert(repository.save_calls == [], "ambiguous page evidence should not save hierarchy")


def test_save_failure_returns_500_without_success() -> None:
    raw_text = "Chapter One\n" + ("current source " * 10)
    loader = _FakeLoader(raw_text=raw_text)
    repository = _FakeRepository(error=RuntimeError("disk full"))
    result = _coordinator(loader, repository).commit_manual_structure_reparse(
        doc_name="Book.txt",
        manual_structure=_plan(source_hash=_hash(raw_text)),
    )

    _assert(not result.success, "save failure must not report success")
    _assert(result.status_code == 500, f"unexpected status: {result.status_code}")
    _assert(result.structured_document_path is None, "failed save must not report a path")
    _assert(result.section_count is None, "failed save should not report committed sections")
    _assert("persistence failed" in (result.error or ""), f"unexpected error: {result.error}")


if __name__ == "__main__":
    test_invalid_projection_does_not_load_source()
    test_missing_source_returns_404_before_commit()
    test_source_hash_mismatch_returns_409_before_commit()
    test_draft_build_rejects_raw_text_out_of_bounds_before_commit()
    test_matching_source_hash_commits_manual_structure_document()
    test_page_range_commit_uses_preparation_page_boundaries()
    test_page_range_commit_rejects_ambiguous_page_boundaries_before_save()
    test_save_failure_returns_500_without_success()
    print("manual structure commit source evidence tests passed")
