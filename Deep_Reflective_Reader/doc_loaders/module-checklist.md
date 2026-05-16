# doc_loaders Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `doc_loaders` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/doc_loaders/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/doc_loaders/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines loader abstraction via `AbstractDocumentLoader.load(doc_name) -> str`.
  Evidence: `Deep_Reflective_Reader/doc_loaders/abstract_document_loader.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Key Files)`
  Notes: Raw-text loading contract is explicit and reusable.

- [x] Implements TXT loader and PDF loader for canonical raw text extraction.
  Evidence: `Deep_Reflective_Reader/doc_loaders/text_document_loader.py; Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Main Responsibilities)`
  Notes: Both loaders normalize `doc_name` with extension handling.

- [x] Implements loader selection through `DocumentLoaderFactory` with extension/path heuristics and historical TXT default.
  Evidence: `Deep_Reflective_Reader/doc_loaders/document_loader_factory.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Factory keeps compatibility default when ambiguity remains.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] <future task>

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
