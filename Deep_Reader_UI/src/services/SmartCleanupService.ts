import type { CleanupOutcome } from "../types/api";
import { validCleanup } from "../shared/cleanupOutcome";
import { RestClient } from "./RestClient";

export interface CleanupConsent {
  operation_id: string;
  confirmation_token: string;
}
export interface CleanupPreflight {
  execution_available: boolean;
  reason: string | null;
  impact: {
    generated_invalidation: "only_if_reading_content_changed";
    preserves_notes: true;
    preserves_manual_edits: true;
    preserves_original: true;
    preserves_structure: true;
  };
  operation_id: string | null;
  confirmation_token: string | null;
  expires_in_seconds: number | null;
}
export interface CleanupOperation {
  operation_id: string;
  phase: "planning" | "staging" | "committing" | "layout" | "finished" | "unknown" | "superseded";
  commit: "not_started" | "not_committed" | "committed" | "unknown";
  outcome: CleanupOutcome | null;
  reason: string | null;
  retry_action: "none" | "reconcile" | "regenerate_cleanup" | "resume_layout";
  reading_ready: boolean | null;
  pending_review_count: number | null;
  durability: "process_local" | "persisted_commit";
}
const reasons = new Set([
  "native_source", "ocr_evidence_missing", "ocr_evidence_mismatch", "extraction_mode_unsupported",
  "capacity_exceeded", "provider_failed", "invalid_plan", "no_safe_items", "some_items_rejected",
  "source_changed", "reading_revision_changed", "storage_failed", "cancelled", "structure_not_ready",
  "cleanup_policy_changed", "capacity_unavailable", "cleanup_service_unavailable", "cleanup_document_missing",
  "cleanup_source_not_ready", "cleanup_identity_changed", "cleanup_in_progress", "cleanup_operation_capacity",
  "cleanup_preflight_unavailable", "cleanup_reconciliation_unavailable", "cleanup_execution_unavailable",
  "cleanup_outcome_unknown", "cleanup_source_changed", "cleanup_operation_superseded", "cleanup_layout_failed",
  "cleanup_confirmation_required", "cleanup_confirmation_invalid", "cleanup_confirmation_expired",
]);
const operationId = (value: unknown): value is string => typeof value === "string" && /^[0-9a-f]{32}$/.test(value);
const token = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9_-]{43}$/.test(value);
const reason = (value: unknown) => value === null || (typeof value === "string" && reasons.has(value));
const integer = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const oneOf = (value: unknown, choices: readonly string[]) => typeof value === "string" && choices.includes(value);
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("cleanup_response_invalid");
  return value as Record<string, unknown>;
}
function requireValid(valid: boolean): asserts valid {
  if (!valid) throw new Error("cleanup_response_invalid");
}
export function parseCleanupPreflight(value: unknown): CleanupPreflight {
  const data = object(value), impact = object(data.impact);
  requireValid(typeof data.execution_available === "boolean" && reason(data.reason)
    && impact.generated_invalidation === "only_if_reading_content_changed"
    && [impact.preserves_notes, impact.preserves_manual_edits, impact.preserves_original, impact.preserves_structure].every(v => v === true));
  requireValid(data.execution_available
    ? data.reason === null && operationId(data.operation_id) && token(data.confirmation_token)
      && integer(data.expires_in_seconds) && data.expires_in_seconds >= 1 && data.expires_in_seconds <= 300
    : data.reason !== null && data.operation_id === null && data.confirmation_token === null && data.expires_in_seconds === null);
  return data as unknown as CleanupPreflight;
}
export function parseCleanupOperation(value: unknown, expectedId: string): CleanupOperation {
  const data = object(value);
  requireValid(operationId(data.operation_id) && data.operation_id === expectedId && reason(data.reason)
    && oneOf(data.phase, ["planning", "staging", "committing", "layout", "finished", "unknown", "superseded"])
    && oneOf(data.commit, ["not_started", "not_committed", "committed", "unknown"])
    && oneOf(data.retry_action, ["none", "reconcile", "regenerate_cleanup", "resume_layout"])
    && oneOf(data.durability, ["process_local", "persisted_commit"])
    && (data.pending_review_count === null || integer(data.pending_review_count)));
  const ready = data.phase === "finished" && data.commit === "committed" ? true : data.phase === "layout" ? false : null;
  requireValid(data.reading_ready === ready);
  if (data.outcome !== null) {
    const outcome = object(data.outcome) as unknown as CleanupOutcome;
    requireValid(validCleanup(outcome) && reason(outcome.reason)
      && (!outcome.reading_content_changed || data.commit === "committed"));
  }
  if (data.phase === "unknown") requireValid(data.commit === "unknown" && data.outcome === null && data.retry_action === "reconcile");
  if (data.phase === "layout") requireValid(data.commit === "committed" && data.outcome !== null && data.retry_action === "resume_layout");
  if (data.phase === "superseded") requireValid(data.commit === "committed" && data.outcome !== null && data.retry_action === "none");
  if (data.phase === "finished") requireValid(data.outcome !== null && oneOf(data.commit, ["committed", "not_committed"])
    && oneOf(data.retry_action, ["none", "regenerate_cleanup"]));
  if (oneOf(data.phase, ["planning", "staging", "committing"])) requireValid(data.outcome === null
    && data.commit === (data.phase === "committing" ? "unknown" : "not_started") && data.retry_action === "none");
  if (ready) requireValid(oneOf(data.retry_action, ["none", "regenerate_cleanup"]));
  return data as unknown as CleanupOperation;
}
export class SmartCleanupService {
  constructor(private readonly client: RestClient) {}
  private path(name: string) {
    if (!name.trim()) throw new Error("cleanup_document_missing");
    return `/documents/${encodeURIComponent(name)}/smart-cleanup`;
  }
  private async request(path: string, options: RequestInit, operation = false) {
    const response = await this.client.requestJsonWithStatus<unknown>(path, options);
    if (!response.ok) {
      // Non-2xx responses may carry authoritative phase/commit information.
      // The caller still validates the complete operation and expected identity.
      if (operation && response.payload && typeof response.payload === "object" && "operation_id" in response.payload) return response.payload;
      const detail = response.payload && typeof response.payload === "object" && "detail" in response.payload ? response.payload.detail : null;
      throw new Error(typeof detail === "string" && reasons.has(detail) ? detail
        : [404, 405, 501].includes(response.status) ? "cleanup_server_unsupported" : "cleanup_service_unavailable");
    }
    return response.payload;
  }
  async preflight(name: string, signal: AbortSignal) {
    return parseCleanupPreflight(await this.request(`${this.path(name)}/preflight`, { method: "POST", body: "{}", signal }));
  }
  async execute(name: string, consent: CleanupConsent, signal: AbortSignal) {
    if (!operationId(consent.operation_id) || !token(consent.confirmation_token)) throw new Error("cleanup_confirmation_invalid");
    return parseCleanupOperation(await this.request(this.path(name), { method: "POST", signal,
      body: JSON.stringify({ operation_id: consent.operation_id, confirmation_token: consent.confirmation_token }) }, true), consent.operation_id);
  }
  async lookup(name: string, id: string, signal: AbortSignal) {
    if (!operationId(id)) throw new Error("cleanup_confirmation_invalid");
    return parseCleanupOperation(await this.request(`${this.path(name)}/${id}`, { method: "GET", signal, cache: "no-store" }, true), id);
  }
}
