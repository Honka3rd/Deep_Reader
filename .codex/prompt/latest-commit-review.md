# External Review of a Deep Reader Task Revision

Perform a read-only code-quality review. Do not modify files, commit, deploy or push.

Inputs from the developer's review packet:

- Task scope and acceptance criteria.
- Base commit: `{{BASE_COMMIT}}`.
- Latest candidate commit: `{{REVIEW_COMMIT}}` (exact SHA).
- Change summary, test/real-data evidence and known limitations.
- Previous review findings and fix mapping, when this is a re-review.

Use the local repository or an exported snapshot/diff that identifies both commits.
The candidate may not have been pushed: do not assume GitHub contains it or review
the remote branch tip instead. If the candidate source, baseline or required
evidence is unavailable, request it and report `NEEDS_INFORMATION`, not approval.

Read applicable AGENTS instructions. Determine affected owners from the cumulative
base-to-candidate diff, then load relevant designs, checklist items and development
standards. Do not load unrelated module documents. Review the committed revision,
not uncommitted work, and verify prior findings against this exact candidate.

Check correctness, regressions, applicable code standards, architecture/ownership,
failure handling, test coverage/evidence and documentation consistency. Inspect
actual code and relevant callers/tests; do not rely only on the implementing
agent's report or on the final fix commit's diff. Report checks not performed and
avoid attributing unrelated pre-existing issues to this task.

Return:

- Reviewed base and candidate SHAs; actual scope and evidence inspected.
- Findings with stable IDs, severity, file/line, governing rule or concrete failure
  scenario, impact, recommended correction and verification expectations.
- Status of each previous finding: resolved, unresolved or needs clarification.
- Checks performed, unavailable evidence, remaining risks and non-blocking notes.
- Verdict: `APPROVED`, `CHANGES_REQUIRED` or `NEEDS_INFORMATION`.

Use `APPROVED` only when the candidate meets the applicable standards and has no
unresolved blocking findings or missing required evidence for this review stage.
Docker deployment acceptance is a subsequent gate, not a prerequisite for this
pre-Docker review. Approval applies only to the stated candidate SHA. Return the
result to the developer for handoff; do not start implementation or Docker work.
