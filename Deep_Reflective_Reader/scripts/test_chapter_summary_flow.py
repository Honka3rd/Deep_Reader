import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.app_DI_config import AppDIConfig
from config.container import ApplicationLookupContainer
from document_preparation.preparation_mode import PreparationMode
from document_structure.structured_document import StructuredDocument, StructuredSection


def _resolve_target_section(
    *,
    document: StructuredDocument,
    section_id: str | None,
    title_prefix: str,
) -> StructuredSection:
    """Resolve one chapter section either by id or by title prefix."""
    if section_id is not None:
        normalized_id = section_id.strip()
        if not normalized_id:
            raise ValueError("section_id cannot be empty string")
        for section in document.sections:
            if section.section_id == normalized_id:
                return section
        raise ValueError(f"section_id '{normalized_id}' not found")

    normalized_prefix = title_prefix.strip().lower()
    if not normalized_prefix:
        raise ValueError("title_prefix cannot be empty")

    for section in document.sections:
        title = (section.title or "").strip().lower()
        if title.startswith(normalized_prefix):
            return section

    raise ValueError(
        "no matching chapter section found "
        f"(prefix='{title_prefix}', total_sections={len(document.sections)})"
    )


def _format_error_context(document: StructuredDocument) -> str:
    """Build short debug context when chapter section cannot be resolved."""
    titles: Iterable[str] = (
        f"{section.section_id}:{(section.title or 'None')}"
        for section in document.sections[:12]
    )
    return ", ".join(titles)


def run(doc_name: str, section_id: str | None, title_prefix: str) -> str:
    """Run minimal chapter-summary flow and return summary text."""
    container = ApplicationLookupContainer.build(AppDIConfig())
    preparation_pipeline = container.document_preparation_pipeline()
    chapter_summary_service = container.chapter_summary_service()

    result = preparation_pipeline.prepare_and_load(
        doc_name=doc_name,
        mode=PreparationMode.BASE,
    )
    document = result.structured_document
    if document is None:
        raise RuntimeError(
            "structured document unavailable after prepare_and_load(base): "
            f"errors={result.assets.errors}"
        )
    if not document.sections:
        raise RuntimeError(
            f"structured document has no sections: document_id='{document.document_id}'"
        )

    try:
        target_section = _resolve_target_section(
            document=document,
            section_id=section_id,
            title_prefix=title_prefix,
        )
    except ValueError as error:
        raise ValueError(
            f"{error}. section_samples=[{_format_error_context(document)}]"
        ) from error

    summary_result = chapter_summary_service.summarize_section(
        document=document,
        section_id=target_section.section_id,
    )
    if not summary_result.success:
        raise RuntimeError(
            "chapter summary generation failed: "
            f"reason={summary_result.reason}"
        )
    summary = (summary_result.payload or "").strip()
    if not summary:
        raise RuntimeError(
            f"empty summary generated for section_id='{target_section.section_id}'"
        )

    print(f"Document Title: {document.title}")
    print(f"Section ID: {target_section.section_id}")
    print(f"Section Title: {target_section.title}")
    print(f"Container Title: {target_section.container_title}")
    print("Summary:")
    print(summary)
    return summary


def main() -> None:
    """CLI entrypoint for minimal chapter-summary flow validation."""
    parser = argparse.ArgumentParser(
        description="Run minimal chapter summary flow from structured document.",
    )
    parser.add_argument(
        "--doc-name",
        required=True,
        help="Document name (same as preparation namespace).",
    )
    parser.add_argument(
        "--section-id",
        default=None,
        help="Optional explicit section id (e.g. section-0).",
    )
    parser.add_argument(
        "--title-prefix",
        default="Chapter",
        help="Fallback title prefix when --section-id is not provided.",
    )
    args = parser.parse_args()
    run(
        doc_name=args.doc_name,
        section_id=args.section_id,
        title_prefix=args.title_prefix,
    )


if __name__ == "__main__":
    main()
