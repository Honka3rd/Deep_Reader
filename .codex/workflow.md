# Deep Reader Delivery Workflow Prototype

## Scope and authority

This is the operational contract for the agent's request-to-delivery workflow,
including developer confirmation and external review. It is a prompt-driven
prototype, not an implemented CI service. It does not change product architecture
or override [AGENTS.md](../AGENTS.md), applicable skills, or module contracts.

Use [the main agent prompt](templates/main-agent-instructions.md) to start work,
[the reload prompt](prompt/reload-main-agent-context.md) to resume, and
[the review prompt](prompt/latest-commit-review.md) for external review.

The full delivery path applies to implementation requests. Explicitly limited
planning, audit, routing, documentation or instruction-maintenance requests stop
at their requested deliverable after appropriate validation. Editing this workflow
does not itself authorize a commit, deployment or push. Read-only skills remain
read-only; their completion returns control to the authorized task.

## 1. Understand the request and route ownership

1. Read applicable AGENTS instructions. Use the request, paths and focused read-only
   code inspection to determine likely change owners and supporting contracts.
2. Use `module-memory-loader` to load only relevant current design sections,
   checklist dependencies and applicable standards. Expand reading when scope
   expands; reuse unchanged context within the same task.
3. State expected behavior, non-goals, acceptance criteria and significant risks.
4. Resolve discoverable facts from the repository. For remaining questions about
   clarity, feasibility, boundaries or conflicting requirements, invoke `grill-me`.
   Ask one question at a time, explain its importance and recommendation, then wait
   for the developer's answer. Do not answer on their behalf or proceed with work
   dependent on an unresolved decision. If the request is clear, continue without
   a mandatory interview or a separate routing approval.
5. Inspect the branch and working tree. Record the task's base commit, intended
   change scope and pre-existing changes before edits; do not absorb unrelated work.

## 2. Define implementation units

Keep small tasks as one unit. Split large tasks into logically independent,
verifiable units with explicit scope, dependencies, documents and acceptance
criteria. Each completed unit should leave a usable, coherent intermediate state.

After each unit of a large task, report its result and wait for the developer's
confirmation before starting the next unit. Silence, elapsed time and a previous
unit's approval do not authorize the next one. After the final unit, proceed to
the external-review stage; this is distinct from the next-unit confirmation gate.

## 3. Update documentation before implementation

1. Update the affected design and checklist before changing implementation. Cover
   behavior, ownership, contracts/data structures, failure handling and acceptance.
2. Replace obsolete current rules instead of leaving contradictory alternatives.
   Mark planned behavior as planned; do not claim it is already implemented.
3. If a new module is justified, create its directory and initialize
   `module-detailed-design.md` and `module-checklist.md`. Empty files are scaffolding
   only: fill the governing design and acceptance criteria before implementation.
   Update the module index and affected architecture contracts where necessary.
   Existing root-file modules retain their registered document paths.
4. Respect Markdown ownership and write authorization. New or status-updated
   checklist items end with `Timestamp: YYYY-MM-DD` using the change date.
   Update `progress.md` only under its explicit authorization/owning skill rules.

## 4. Implement and validate each unit

1. Implement against the documented contract and applicable development standards.
2. Run relevant static checks, unit/integration tests and representative real-data
   tests, including meaningful failure and boundary cases.
3. Prefer real-data copies and isolated test storage. Record sample provenance,
   commands, expected/actual results and evidence. Distinguish real input data,
   mocked providers, real external services and semantic-quality assessment;
   one does not prove the others. If acceptance requires paid LLM/service calls,
   use established authorization and limits; ask for missing required limits
   rather than assuming unlimited calls. Reuse valid evidence where appropriate.
4. Fix failures and rerun affected checks. Missing required checks remain unresolved;
   do not label them passed or silently substitute mock results.
5. Reconcile documents with the implementation and evidence. Only mark supported
   checklist items complete. Report actual inspected/modified documents and reasons.
6. Use `implementation-validation` within the identified scope. This self-validation
   does not substitute for external review. For a large task, enforce the next-unit
   confirmation gate before implementing another unit.

## 5. Commit and obtain external code-quality review

1. Once all units and required checks are complete, inspect the task diff and stage
   only attributable changes. Preserve pre-existing work. If mixed changes cannot
   be separated reliably, resolve the scope with the developer before committing.
   Check for secrets, unintended data and generated artifacts.
2. Create a local review commit on the task branch; do not push yet. Detect the
   actual branch and remote instead of using a historical hardcoded branch. Use or
   create an appropriate task branch without resetting existing work. Do not amend
   or rewrite earlier review commits; fixes receive new commits.
   After explicit staging and diff inspection, use
   `scripts/commit_review.sh --message-file <file>`. Write the exact commit message
   to that file, preserving multiline text. Add `--dry-run` to check the staged
   scope without committing. The script commits only the existing index, rejects
   empty/conflicted changes, detached HEAD and unfinished merge/rebase/cherry-pick/
   revert operations, and preserves normal Git hooks/signing. It does not stage,
   amend, push, run project tests, scan for secrets or grant external approval;
   those remain workflow duties.
3. Provide the base commit, latest commit SHA, task diff scope, change summary,
   test evidence and known limitations. Review must cover the cumulative task
   changes, not only the last fix commit.
4. Prompt the developer: "Please externally review the latest commit identified
   above and return the audit result. Docker work is blocked until this exact
   revision is approved." The review may use the local repository or an exported
   commit/diff snapshot; an unpushed commit is not available through GitHub alone.
5. Stop and wait for the developer's audit result. Missing/incomplete results,
   results for another SHA, and the implementing agent's own checks are not approval.
6. Map each finding to the applicable standard/contract and code evidence. Clarify
   ambiguous or conflicting findings. Update design where needed, implement fixes,
   rerun relevant tests (including affected real-data cases), and reconcile docs.
7. Create a new commit and provide its SHA, finding-to-fix mapping and test results.
   Request external review again and wait. Repeat until the latest commit is
   explicitly approved with no unresolved blocking findings. Never waive this gate
   because of elapsed time, iteration count or apparent smallness of a fix.
8. Record the approved SHA and audit conclusion. Only then proceed to Docker work.

## 6. Rebuild the local Docker runtime

Run `scripts/start_local_dev.sh --backend-only` from the approved checkout to rebuild
Docker through the existing environment loader, reviewed migrations, health waits
and connectivity check. This preserves an already running host UI; omit
`--backend-only` only when also starting the UI. Do not copy this sequence into
ad hoc Compose commands or temporary deployment scripts. Read-only inspection,
backups and acceptance probes may use separate commands.

Confirm that build inputs match the approved revision, including no untracked
source files leaking into the Docker context. Use an isolated checkout if necessary;
preserve the existing Compose project identity and absolute data/volume destinations
through the supported local environment settings. Do not reset unrelated work or
silently create a second database/empty data directory. If the existing script
cannot safely support the required setup, improve it through the normal change and
external-review loop before deployment, rather than bypassing it.

Preserve volumes and existing data; destructive cleanup is not a normal rebuild
step. Check service health, configuration loading and required migrations. Verify
the running services actually contain the approved changes rather than an old image.

## 7. Validate the Docker deployment

Exercise the relevant UI/API flows through the actual local Docker deployment,
including representative real-data cases. Inspect logs and persisted results to
verify behavior and record the revision, deployment identity and acceptance evidence.

If acceptance fails, diagnose and fix it, update relevant documents, run affected
tests, create a new commit, and return to external review before rebuilding and
revalidating. Any subsequent change to the delivery revision requires renewed
review; runtime-affecting changes also require rebuilding and rerunning affected
Docker checks. Documentation-only changes may reuse unaffected runtime evidence
with an explicit explanation linking it to the new revision.

Required acceptance failures or unavailable checks block remote delivery.

## 8. Push the verified task branch

1. Check that the intended branch tip matches the reviewed revision, has applicable
   Docker acceptance evidence, and includes only the intended task commits relative
   to the remote. Resolve unrelated outgoing commits rather than pushing them
   accidentally. Check for uncommitted task changes that would invalidate evidence.
2. Push the verified commits to the intended remote task branch. Do not create a PR,
   merge into the default branch, force push, or perform production deployment.
   Do not create an empty final commit when the verified commits already exist.
3. Verify the remote branch SHA matches the intended local tip. A failed/rejected
   push is not delivery. Preserve the work and report the failure; do not overwrite
   remote history. Changes introduced by conflict resolution return to validation
   and external review, with affected Docker checks rerun.

## 9. Report and resume

Report behavior/modules delivered, inspected and modified documents with reasons,
test and real-data evidence, approved review SHA/conclusion, Docker results,
remaining limitations, and the pushed commit/remote branch verification.

At an interruption or human gate, leave a concise checkpoint in the handoff:

- Request, scope, base commit, branch and current stage/unit.
- Completed work and validation evidence; pre-existing work to preserve.
- Pending findings, unresolved questions and exact next action.
- Latest review SHA, approved SHA if any, and Docker evidence revision.
- Whether waiting for a next-unit confirmation or external audit result.

On resume, verify the checkpoint against repository/runtime state and continue at
the first unmet gate. Do not infer approval from silence or replay successful
expensive operations without a reason. This prototype requires no new checkpoint
database, CI platform, automatic external-review service or PR automation.
