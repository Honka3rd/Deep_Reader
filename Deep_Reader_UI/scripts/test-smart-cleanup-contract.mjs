import assert from "node:assert/strict";
import { createServer } from "vite";
const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { parseCleanupPreflight, parseCleanupOperation, SmartCleanupService } = await server.ssrLoadModule("/src/services/SmartCleanupService.ts");
  const id = "a".repeat(32), token = "b".repeat(43);
  const ready = { execution_available: true, reason: null, operation_id: id, confirmation_token: token, expires_in_seconds: 60,
    impact: { generated_invalidation: "only_if_reading_content_changed", preserves_notes: true,
      preserves_original: true, preserves_manual_edits: true, preserves_structure: true } };
  assert.deepEqual(parseCleanupPreflight(ready), ready);
  let rejected = 0;
  for (const patch of [{ execution_available: "true" }, { reason: "provider private content" }, { operation_id: "wrong" },
    { confirmation_token: "wrong" }, { expires_in_seconds: 301 }, { expires_in_seconds: 0 }, { expires_in_seconds: 1.5 },
    { impact: {} }, { impact: { ...ready.impact, preserves_notes: false } }, { execution_available: false }]) {
    assert.throws(() => parseCleanupPreflight({ ...ready, ...patch })); rejected++;
  }
  for (const value of [null, [], true, { execution_available: true }]) {
    assert.throws(() => parseCleanupPreflight(value)); rejected++;
  }
  const result = { operation_id: id, phase: "finished", commit: "committed", reason: null, retry_action: "none",
    reading_ready: true, pending_review_count: 0, durability: "persisted_commit", outcome: {
      status: "completed", proposed_count: 1, applied_count: 1, skipped_count: 0, reading_content_changed: true,
      reading_revision: 1, reason: null } };
  assert.deepEqual(parseCleanupOperation(result, id), result);
  assert.equal(parseCleanupOperation({ ...result, retry_action: "regenerate_cleanup", outcome: {
    ...result.outcome, status: "partial", proposed_count: 2, skipped_count: 1, reason: "some_items_rejected" } }, id).outcome.status, "partial");
  for (const patch of [{ operation_id: "c".repeat(32) }, { phase: "complete" }, { commit: "unknown" },
    { phase: ["finished"] }, { commit: ["committed"] }, { durability: ["persisted_commit"] }, { retry_action: ["none"] },
    { reading_ready: false }, { pending_review_count: -1 }, { pending_review_count: "0" }, { outcome: null },
    { durability: "unknown" }, { retry_action: "resume_layout" }, { outcome: {} },
    { outcome: { ...result.outcome, proposed_count: 2 } }, { outcome: { ...result.outcome, reason: "PRIVATE" } },
    { outcome: { ...result.outcome, reading_content_changed: false } }]) {
    assert.throws(() => parseCleanupOperation({ ...result, ...patch }, id)); rejected++;
  }
  assert.equal(parseCleanupOperation({ ...result, phase: "unknown", commit: "unknown", outcome: null,
    reading_ready: null, retry_action: "reconcile" }, id).phase, "unknown");
  const layout = { ...result, phase: "layout", reading_ready: false, retry_action: "resume_layout" };
  assert.equal(parseCleanupOperation(layout, id).phase, "layout");
  for (const value of [{ ...layout, retry_action: "none" }, { ...layout, commit: "unknown" },
    { ...result, phase: "superseded", reading_ready: null, retry_action: "resume_layout" },
    { ...result, phase: "planning", reading_ready: null, outcome: null },
    { ...result, phase: "finished", commit: "not_committed", reading_ready: null, outcome: null }]) {
    assert.throws(() => parseCleanupOperation(value, id)); rejected++;
  }
  const calls = [], abort = new AbortController();
  let reply = { ok: true, payload: ready, status: 200 };
  const service = new SmartCleanupService({ requestJsonWithStatus: async (...args) => { calls.push(args); return reply; } });
  await service.preflight("a/b", abort.signal);
  assert.equal(calls[0][0], "/documents/a%2Fb/smart-cleanup/preflight");
  assert.equal(calls[0][1].signal, abort.signal);
  assert.deepEqual(JSON.parse(calls[0][1].body), {});
  reply = { ok: true, payload: result, status: 200 };
  await service.execute("a/b", { operation_id: id, confirmation_token: token, unsafe: "ignored" }, abort.signal);
  assert.deepEqual(JSON.parse(calls[1][1].body), { operation_id: id, confirmation_token: token });
  reply = { ok: false, status: 503, payload: { ...result, phase: "unknown", commit: "unknown", outcome: null,
    reading_ready: null, retry_action: "reconcile", reason: "cleanup_reconciliation_unavailable" } };
  assert.equal((await service.lookup("book", id, abort.signal)).phase, "unknown");
  assert.equal(calls[2][0], `/documents/book/smart-cleanup/${id}`);
  assert.equal(calls[2][1].method, "GET");
  assert.equal(calls[2][1].body, undefined);
  assert.equal(calls[2][1].cache, "no-store");
  reply = { ok: false, status: 409, payload: { ...result, commit: "not_committed", reading_ready: null,
    reason: "reading_revision_changed", outcome: { ...result.outcome, status: "skipped", reason: "reading_revision_changed",
      reading_content_changed: false, applied_count: 0, proposed_count: 0, reading_revision: 0 } } };
  assert.equal((await service.execute("book", { operation_id: id, confirmation_token: token }, abort.signal)).reason, "reading_revision_changed");
  await assert.rejects(service.preflight(" ", abort.signal));
  await assert.rejects(service.execute("book", { operation_id: "invalid", confirmation_token: token }, abort.signal));
  for (const [status, payload, error] of [[404, { detail: "Not Found" }, "cleanup_server_unsupported"],
    [503, { detail: "PRIVATE" }, "cleanup_service_unavailable"], [409, { detail: "cleanup_in_progress" }, "cleanup_in_progress"]]) {
    reply = { ok: false, payload, status };
    await assert.rejects(service.preflight("book", abort.signal), { message: error });
  }
  const recovery = await server.ssrLoadModule("/src/features/smart-cleanup/recovery.ts");
  const storage = new Map();
  globalThis.sessionStorage = { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  try {
    recovery.rememberCleanup("book", { operation_id: id, confirmation_token: token, result: "not retained" });
    assert.deepEqual(recovery.recoverCleanup("book"), { operation_id: id, confirmation_token: token });
    assert.equal(recovery.recoverCleanup("another"), null);
    recovery.forgetCleanup("book", "c".repeat(32));
    assert.ok(recovery.recoverCleanup("book"));
    recovery.forgetCleanup("book", id);
    assert.equal(recovery.recoverCleanup("book"), null);
    storage.set("deep-reader.smart-cleanup:book", "broken json");
    assert.throws(() => recovery.recoverCleanup("book"), { message: "cleanup_recovery_invalid" });
  } finally { delete globalThis.sessionStorage; }
  console.log(`PASS cleanup adapter: ${rejected} malformed responses rejected; valid/partial/unknown, narrow requests, identity and safe transport errors`);
} finally { await server.close(); }
