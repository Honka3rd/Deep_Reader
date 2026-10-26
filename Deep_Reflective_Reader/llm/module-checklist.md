# llm Checklist

## Offline Preflight Follow-up (2026-10-24)

Shared WAIT track: 5/5 implemented and accepted (100%); backend 4/4, UI 1/1.
WAIT-01..05 have evidence in their owner checklists. Existing SCLEAN
completion remains historical scoped evidence. Runtime deployment is not claimed.
Timestamp: 2026-10-24
Design: [incident and contract](module-detailed-design.md#offline-preflight-follow-up-2026-10-24).
Checkboxes live only with their owners; this table is the execution index.

| Order | Task | Owner | Scope | Dependency |
| --- | --- | --- | --- | --- |
| 1 | WAIT-01 | [UI](../../Deep_Reader_UI/module-checklist.md#request-wait-policy-follow-up-2026-10-24) | Remove frontend fixed request deadlines, retain lifecycle/recovery | None |
| 2 | WAIT-02 | llm | Shared offline-only token counting | None |
| 3 | WAIT-03 | [scripts](../scripts/module-checklist.md#offline-preflight-follow-up-2026-10-24) | Package/verify tokenizer assets in image and local setup | WAIT-02 |
| 4 | WAIT-04 | [app](../app/module-checklist.md#offline-preflight-follow-up-2026-10-24) | Correlated preflight stage/outcome timings | WAIT-02 |
| 5 | WAIT-05 | [scripts](../scripts/module-checklist.md#offline-preflight-follow-up-2026-10-24) | Cold/offline/real-book and browser acceptance | WAIT-01..04 |

- [x] WAIT-02: Make shared token counting offline-only with explicit safe fallback.
  Priority: P0. Owner: llm; document_structure supports existing budget/fingerprint consumers.
  Acceptance: Verified model-matched local encoding only; no request/import-time download, wrong-encoding fallback, zero-count acceptance or prompt truncation. Missing/corrupt/unsupported assets return unavailable promptly; existing labelled conservative capacity decision remains fail-closed where necessary. Preflight/dispatch identity changes require fresh consent. Add private-data-safe local initialization/count timings.
  Evidence required: Network-denied cold/warm/missing/corrupt/unsupported/concurrent initialization tests; special-token text and full prompt/schema accounting; unchanged model limits/output reserve; zero completion calls and prompt/source-free logs. Tests accompany this task rather than wait for final acceptance.
  Evidence: offline_token_counter owns the five-model/o200k manifest, hash/size-verified local reads, pinned public Encoding recipe and synchronized shared successful cache. OpenAILLMProvider delegates without registry download. Missing/corrupt/mismatched/unavailable paths return None, preserving the labelled conservative budget fallback. Internal recipe/hash/library identity flows through SemanticPlanProvider into existing split/cleanup consent fingerprints, without new API exposure or changed capacity policy. Safe stage durations distinguish verification, initialization and counting.
  Validation: 129 tests pass, including 13 new counter cases and opt-in verified real-asset/immutable-book checks. Fresh processes with denied network and empty temporary cache, warm/concurrent counts, missing/corrupt/unreadable assets, absent dependency, unsupported mapping, special-token/multilingual parity, full schema accounting and changed identity/fallback refusal pass. Madame Bovary/Red Roulette complete accepted-text counting preserves manifest hashes; zero model calls. No production mutation/deployment; default asset packaging remains WAIT-03, app spans WAIT-04 and full incident-book/container acceptance WAIT-05.
  Timestamp: 2026-10-24

## Full-Source Enhancement Follow-up

- [x] FULL-01: Separate verified context/input/output capacity and migrate budget consumers.
  Acceptance: Preserve verified gpt-4.1-mini context=1047576/output=32768; label current effective output=3276 as policy. Add explicit context-window versus optional input-only semantics, audit all legacy max_input_tokens consumers, avoid double output reservation, match tokenizer identity and fail closed on unavailable capacity. Verify other models independently before migrating their constants.
  Evidence needed: Official source recorded in design; capability/config/normalizer boundary tests and migration compatibility tests. No provider switch or assumed account quota.
  Delivery: Explicit context/input/output fields, shared input ceiling and QA combined-budget adapter; all in-repository consumers/fixtures audited and migrated. Provider, retrieval and planner logs distinguish limits; planner budget v2 retains fail-closed behavior. Other model entries independently verified against official pages linked in design.
  Validation: 66 unittest cases including 9 new capacity cases and 10 standalone regression scripts passed; git diff --check passed. No paid LLM or deployment. Full-source assembly and frontend preflight remain FULL-02..05.
  Timestamp: 2026-10-20

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `llm` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/llm/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/llm/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines LLM provider abstraction contract for completion and capability reporting.
  Evidence: `Deep_Reflective_Reader/llm/llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Key Files)`
  Notes: Shared provider contract is used across multiple modules.

- [x] Implements OpenAI-backed provider with model-capability mapping and endpoint routing.
  Evidence: `Deep_Reflective_Reader/llm/openai_llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Main Responsibilities)`
  Notes: Provider selects responses vs chat-completions based on model capabilities.

- [x] Implements capability-aware prompt text normalization fallback helper.
  Evidence: `Deep_Reflective_Reader/llm/llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Main Flows)`
  Notes: Normalization computes safe excerpts with fallback when capabilities unavailable.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Expose model capability metadata for reading interaction context gating
  Evidence needed: interaction orchestration can determine full-target versus compact-context mode using provider capability metadata.
  Notes: Provider remains a capability/completion boundary, not a target resolver or artifact owner.
  Timestamp: 2026-10-05
- [ ] Preserve provider boundary during structured interaction generation
  Evidence needed: provider returns completion output while strict JSON validation, artifact persistence, and permission/cost gating stay outside `llm/`.
  Notes: This keeps LLM integration reusable across analysis, quiz, and critical-thinking flows.
  Timestamp: 2026-10-05

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
