# bundle_factory.py Checklist

## Content Read Performance (2026-10-21)

- [x] CONTENT-03: Make runtime bundle profile access reuse-only and preserve profiles during index rebuilds.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: BundleFactory no longer builds or clears profiles; explicit readiness failures preserve bytes. test_content_read_recovery covers normal reuse, corrupt/missing profiles and index cleanup with zero profile builder calls.
  Timestamp: 2026-10-21

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `bundle_factory.py` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/bundle_factory.module-detailed-design.md`
- `Deep_Reflective_Reader/bundle_factory.py`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements bundle cache lifecycle (put, evict, invalidate, clear) for document-scoped runtime bundles.
  Evidence: `Deep_Reflective_Reader/bundle_factory.py; Deep_Reflective_Reader/bundle_factory.module-detailed-design.md (Main Responsibilities)`
  Notes: Factory keeps bounded in-memory cache with eviction behavior.

- [x] Implements reuse-only profile readiness; missing/corrupt profile requires explicit preparation/reparse.
  Evidence: `Deep_Reflective_Reader/bundle_factory.py; Deep_Reflective_Reader/bundle_factory.module-detailed-design.md (Main Flows)`
  Notes: Profile is guaranteed before final bundle save/load return.
  Timestamp: 2026-10-21

- [x] Implements index readiness with fingerprint matching and legacy-record-schema rebuild guard.
  Evidence: `Deep_Reflective_Reader/bundle_factory.py; Deep_Reflective_Reader/bundle_factory.module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Rebuild is triggered when schema compatibility checks fail.

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
