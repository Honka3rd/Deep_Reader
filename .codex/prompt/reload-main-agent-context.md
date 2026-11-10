# Resume Deep Reader Delivery Work

Read applicable `AGENTS.md`, `.codex/templates/main-agent-instructions.md` and
`.codex/workflow.md`. Use the repository's current instructions and the user's
active request; do not restore the retired Main Agent V2 stage-by-stage handoff.

Inspect the actual branch, HEAD and working-tree changes. Recover the latest task
checkpoint: scope/base commit, current unit/stage, pre-existing work, test evidence,
latest reviewed SHA, approved SHA and any Docker acceptance revision. Load only
the module contracts relevant to the next action, following scope-first reading.

Briefly report the current state, any discrepancy and the first unmet gate.
Continue already-authorized work from there. Preserve pending next-unit confirmation
and external-review gates; never infer approval from an interruption or silence.
If a necessary checkpoint decision cannot be recovered, ask for that missing
information only. Do not repeat completed expensive operations without cause.

If the request is context refresh only, report the state and stop. Do not begin
implementation, commit, Docker work or remote push solely because this prompt was
loaded. Never assume a fixed historical branch or unverified GitHub snapshot.
