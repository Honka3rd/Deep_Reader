import json

from document_structure.structured_document import StructuredDocument, StructuredSection
from section_tasks.task_unit_resolver import TaskUnitResolver


def build_single_huge_section_document() -> StructuredDocument:
    """Build one structured document that only has one giant section."""
    giant_content = "\n\n".join(
        [
            "Paragraph A " + ("alpha " * 120),
            "Paragraph B " + ("beta " * 140),
            "Paragraph C " + ("gamma " * 150),
        ]
    )
    return StructuredDocument(
        document_id="doc-single-huge",
        title="Single Huge Section Demo",
        source_path=None,
        language="en",
        raw_text=giant_content,
        sections=[
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="Main Body",
                level=1,
                content=giant_content,
                char_start=0,
                char_end=len(giant_content),
                container_title=None,
            )
        ],
    )


def build_many_tiny_sections_document() -> StructuredDocument:
    """Build one structured document that has many tiny adjacent sections."""
    sections = [
        StructuredSection(
            section_id="s-1",
            section_index=0,
            title="Chapter One",
            level=1,
            content="Emma arrives in town.",
            char_start=0,
            char_end=21,
            container_title="Part I",
        ),
        StructuredSection(
            section_id="s-2",
            section_index=1,
            title="Chapter One (cont.)",
            level=1,
            content="She meets Charles.",
            char_start=22,
            char_end=41,
            container_title="Part I",
        ),
        StructuredSection(
            section_id="s-3",
            section_index=2,
            title="Chapter Two",
            level=1,
            content="They discuss daily life and expectations in detail. " * 5,
            char_start=42,
            char_end=280,
            container_title="Part I",
        ),
    ]
    return StructuredDocument(
        document_id="doc-many-tiny",
        title="Many Tiny Sections Demo",
        source_path=None,
        language="en",
        raw_text="\n".join(section.content for section in sections),
        sections=sections,
    )


def main() -> None:
    resolver = TaskUnitResolver(
        task_unit_min_chars=80,
        task_unit_max_chars=600,
    )

    single_huge_doc = build_single_huge_section_document()
    huge_units = resolver.resolve(single_huge_doc)

    tiny_doc = build_many_tiny_sections_document()
    tiny_units = resolver.resolve(tiny_doc)

    payload = {
        "single_huge_section_units": [unit.to_dict() for unit in huge_units],
        "many_tiny_sections_units": [unit.to_dict() for unit in tiny_units],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
