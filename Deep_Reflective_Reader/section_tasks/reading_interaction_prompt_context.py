from __future__ import annotations

from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)


NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION = (
    "Do not use web search, external research, outside facts, or unstated background "
    "knowledge unless an explicit external_research mode is provided."
)


def build_evidence_priority_context_text(
    request: ReadingInteractionRequest,
    *,
    secondary_context_label: str = "Secondary Artifact Signals",
) -> str:
    """Build bounded prompt context with selected target text as primary evidence."""
    target = request.target
    lines = [
        "Evidence priority rules:",
        "- Treat the Primary Source as the only direct textual evidence for this task.",
        "- Use Ancestor Summaries and Document Profile only to orient local context.",
        "- Do not let summaries override or replace the selected source text.",
        "- If the Primary Source does not support a claim, mark it ambiguous or insufficient.",
        f"- {NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION}",
        "",
        "Target Identity:",
        f"Document title: {target.document_title}",
        f"Target level: {target.target_level}",
        f"Target id: {target.target_id}",
    ]
    if target.chapter_id:
        lines.append(f"Chapter id: {target.chapter_id}")
    if target.section_id:
        lines.append(f"Section id: {target.section_id}")
    if target.task_unit_id:
        lines.append(f"Task unit id: {target.task_unit_id}")
    if target.content_block_id:
        lines.append(f"Content block id: {target.content_block_id}")

    lines.extend(
        [
            "",
            f"Primary Source ({_primary_source_label(target.target_level)}):",
            target.content,
        ]
    )

    ancestor_context = request.ancestor_context
    if ancestor_context is not None and ancestor_context.summaries:
        lines.extend(["", "Ancestor Summaries:"])
        for index, summary in enumerate(ancestor_context.summaries, start=1):
            reference = summary.to_prompt_reference()
            lines.extend(
                [
                    f"{index}. target_level: {reference['target_level']}",
                    f"   target_id: {reference['target_id']}",
                ]
            )
            if reference.get("title"):
                lines.append(f"   title: {reference['title']}")
            lines.append(f"   schema_version: {reference['schema_version']}")
            if reference.get("prompt_instruction_version"):
                lines.append(
                    "   prompt_instruction_version: "
                    f"{reference['prompt_instruction_version']}"
                )
            if reference.get("source_structure_version") is not None:
                lines.append(
                    "   source_structure_version: "
                    f"{reference['source_structure_version']}"
                )
            if reference.get("source_hash"):
                lines.append(f"   source_hash: {reference['source_hash']}")
            if reference.get("generated_at"):
                lines.append(f"   generated_at: {reference['generated_at']}")
            lines.append(f"   summary: {reference['summary']}")

    document_profile_summary = _metadata_text(
        request.context_metadata,
        "document_profile_summary",
        "document_summary",
        "profile_summary",
    )
    document_language = _metadata_text(
        request.context_metadata,
        "document_language",
        "language",
    ) or target.document_language
    if document_profile_summary or document_language:
        lines.extend(["", "Document Profile:"])
        if document_profile_summary:
            lines.append(document_profile_summary)
        if document_language:
            lines.append(f"Document language: {document_language}")

    if request.external_research_context is not None:
        external_context = request.external_research_context
        lines.extend(
            [
                "",
                "External Research Mode:",
                "Explicit external_research mode is enabled.",
                f"Usage scope: {external_context.usage_scope}",
                "External material is background context only and must not override Primary Source evidence.",
                "Every external claim must remain source-labelled.",
                "External Sources:",
            ]
        )
        for source in external_context.sources:
            lines.append(
                f"- [{source.label}] {source.source_type}: "
                f"{source.title or source.source_id}"
            )
            if source.url:
                lines.append(f"  url: {source.url}")
            if source.retrieved_at:
                lines.append(f"  retrieved_at: {source.retrieved_at}")

    if request.secondary_context:
        lines.extend(["", f"{secondary_context_label}:", request.secondary_context])

    return "\n".join(lines)


def _primary_source_label(target_level: str) -> str:
    if target_level == "content_block":
        return "selected content_block text"
    return "selected target text"


def _metadata_text(metadata: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None
