import { RestClient } from "./RestClient";
import { isBook, type BookPreferences, type LearningSelection } from "./ReadingPreferencesService";

export type LearningComponent = "translation" | "hints";
export type OperationState = "claimed" | "running" | "partial" | "succeeded" | "failed" | "unknown" | "stale";
export interface LearningHandle { doc_name: string; operation_id: string }
export interface Sentence { sentence_id: string | null; start: number; end: number; text: string }
export interface Hint { sentence_id: string; span: { start: number; end: number }; source_form: string; language_code: string;
  kind: "word" | "phrase" | "idiom"; meaning: string; lemma: string | null; part_of_speech: string | null }
interface SavedBase { result_revision: number; source_language: string; explanation_language: string }
export interface HintAssessment { candidate_count: number; accepted_count: number; filtered_count: number;
  rejected_count: number; omitted_sentences: number; issues: { code: string; count: number }[] }
export type LearningSaved = (SavedBase & { component: "translation"; rows: { sentence_id: string; translated_text: string }[] })
  | (SavedBase & { component: "hints"; proficiency: LearningSelection; items: Hint[]; assessment?: HintAssessment | null });
export interface LearningView {
  component: LearningComponent; state: "missing" | "ready" | "pending" | "partial" | "failed" | "stale" | "unknown" | "unavailable";
  saved: LearningSaved | null; operation: LearningHandle | null; operation_state: OperationState | null;
}
export interface LearningBlock { content_block_id: string; source_token: string; source_length: number; sentences: Sentence[] | null; components: LearningView[] }
export interface LearningRead { doc_name: string; settings: BookPreferences; blocks: LearningBlock[] }
export interface LearningCommand {
  doc_name: string; content_block_id: string; component: LearningComponent; command_id: string; expected_source_token: string;
  expected_workspace_revision: number; expected_book_revision: number; intent: "get_or_generate" | "regenerate" | "retry";
  retry_operation?: LearningHandle; acknowledge_possible_duplicate_cost?: boolean;
}
export interface Submission {
  doc_name: string; command_id: string; outcome: "claimed" | "reuse" | "coalesced" | "reconcile" | "retry_required" | "no_content" | "denied" | "unavailable"
    | "unknown" | "missing" | "superseded" | "stale" | "failed" | "succeeded" | "resuming";
  operation: LearningHandle | null; replay: boolean; reason: string | null; result_revision: number | null;
}
export interface Operation {
  handle: LearningHandle; state: OperationState | "missing"; durable_state: OperationState | null; component: LearningComponent | null;
  content_block_id: string | null; source_token: string | null; source_current: boolean; settings_current: boolean; owns_current_slot: boolean;
  completed_batches: number; total_batches: number; result_revision: number | null; outcome_uncertain: boolean; failure_code: string | null;
}
export interface CommandReceipt { doc_name: string; command_id: string; found: boolean; receipt: Submission | null }
export interface PartialRead { handle: LearningHandle; component: LearningComponent; source_token: string; offset: number;
  completed_batches: number; total_batches: number; next_offset: number | null;
  batches: ({ component: "translation"; batch_index: number; rows: { sentence_id: string; translated_text: string }[] }
    | { component: "hints"; batch_index: number; items: Hint[]; assessment?: HintAssessment | null })[] }
export class LearningFailure extends Error {
  constructor(readonly kind: "read" | "rejected" | "unknown", readonly code = "") {
    super(kind === "unknown" ? "The outcome is unknown. Check progress before retrying."
      : kind === "rejected" ? "This request could not be accepted. Read the current page again before retrying."
      : "Learning data could not be verified. Check progress to read it again.");
  }
}
const obj = (v: unknown): v is Record<string, any> => !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && !!v.trim();
const nat = (v: unknown): v is number => Number.isSafeInteger(v) && Number(v) >= 0;
const positive = (v: unknown): v is number => nat(v) && v > 0;
const token = (v: unknown): v is string => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
const component = (v: unknown): v is LearningComponent => v === "translation" || v === "hints";
const states = ["claimed", "running", "partial", "succeeded", "failed", "unknown", "stale"];
const handle = (v: unknown): v is LearningHandle => obj(v) && text(v.doc_name) && text(v.operation_id);
const equalHandle = (a: LearningHandle, b: LearningHandle) => a.doc_name === b.doc_name && a.operation_id === b.operation_id;
function validAssessment(v: unknown, retained: number): boolean {
  if (v == null) return true; // Saved legacy results have no filtering assessment.
  return obj(v) && [v.candidate_count, v.accepted_count, v.filtered_count, v.rejected_count, v.omitted_sentences].every(nat)
    && v.accepted_count === retained && v.candidate_count === v.accepted_count + v.filtered_count + v.rejected_count
    && Array.isArray(v.issues) && v.issues.every(i => obj(i) && text(i.code) && positive(i.count))
    && new Set(v.issues.map(i => i.code)).size === v.issues.length
    && (!(v.rejected_count || v.omitted_sentences) || v.issues.length > 0);
}
function rows(v: unknown): boolean { return Array.isArray(v) && v.every(r => obj(r) && text(r.sentence_id) && text(r.translated_text)) && new Set(v.map(r => r.sentence_id)).size === v.length; }
function hints(v: unknown): v is Hint[] {
  return Array.isArray(v) && v.every(i => obj(i) && text(i.sentence_id) && obj(i.span) && nat(i.span.start) && positive(i.span.end)
    && i.span.end > i.span.start && [i.source_form, i.language_code, i.meaning].every(text) && ["word", "phrase", "idiom"].includes(i.kind)
    && (i.lemma === null || text(i.lemma)) && (i.part_of_speech === null || text(i.part_of_speech)))
    && new Set(v.map(i => JSON.stringify([i.sentence_id, i.span.start, i.span.end, i.kind]))).size === v.length;
}
export function sameSettings(a: BookPreferences, b: BookPreferences): boolean {
  const selection = (v: BookPreferences) => { const s = v.saved_settings.find(s => s.mode_id === "language_learning")?.selection;
    return s ? [s.version, s.language_code, s.scheme_id, s.scheme_version, s.level_id] : null; };
  return JSON.stringify([a.doc_name, a.revision, a.workspace.revision, a.workspace.native_language, a.active_mode, a.effective_mode, a.source_language, selection(a)])
    === JSON.stringify([b.doc_name, b.revision, b.workspace.revision, b.workspace.native_language, b.active_mode, b.effective_mode, b.source_language, selection(b)]);
}
export function validRead(v: unknown): v is LearningRead {
  if (!obj(v) || !text(v.doc_name) || !isBook(v.settings) || v.settings.doc_name !== v.doc_name || !Array.isArray(v.blocks) || v.blocks.length > 50
    || new Set(v.blocks.map(b => b?.content_block_id)).size !== v.blocks.length) return false;
  const settings = v.settings, selection = settings.saved_settings.find(s => s.mode_id === "language_learning")?.selection;
  return v.blocks.every(b => {
    if (!obj(b) || !text(b.content_block_id) || !token(b.source_token) || !nat(b.source_length) || !Array.isArray(b.components)) return false;
    if (settings.effective_mode !== "language_learning") return b.sentences === null && !b.components.length;
    if (!Array.isArray(b.sentences) || b.components.length !== 2 || new Set(b.components.map(c => c?.component)).size !== 2) return false;
    let end = 0; const spans = new Map<string, Sentence>();
    for (const s of b.sentences) {
      if (!obj(s) || !nat(s.start) || !positive(s.end) || s.start !== end || s.end <= s.start || typeof s.text !== "string"
        || Array.from(s.text).length !== s.end - s.start || (s.sentence_id === null ? !!s.text.trim() : !text(s.sentence_id) || !s.text.trim() || spans.has(s.sentence_id))) return false;
      end = s.end; if (s.sentence_id !== null) spans.set(s.sentence_id, s as Sentence);
    }
    if (end !== b.source_length) return false;
    const source = Array.from(b.sentences.map(s => s.text).join(""));
    return b.components.every(c => {
      if (!obj(c) || !component(c.component) || !["missing", "ready", "pending", "partial", "failed", "stale", "unknown", "unavailable"].includes(c.state)
        || (c.operation === null ? c.operation_state !== null : !handle(c.operation) || c.operation.doc_name !== v.doc_name || !states.includes(c.operation_state))
        || (c.state === "ready") !== (c.saved !== null)) return false;
      if (c.saved === null) {
        const expected = c.operation_state === "claimed" || c.operation_state === "running" ? "pending" : c.operation_state === "succeeded" ? "unavailable" : c.operation_state;
        return c.operation ? c.state === expected : ["missing", "stale", "unavailable"].includes(c.state);
      }
      const saved = c.saved;
      if (!obj(saved) || saved.component !== c.component || !positive(saved.result_revision)
        || saved.source_language !== settings.source_language || saved.explanation_language !== settings.workspace.native_language) return false;
      if (saved.component === "translation") return rows(saved.rows) && JSON.stringify(saved.rows.map((r: { sentence_id: string }) => r.sentence_id)) === JSON.stringify([...spans.keys()]);
      const p = saved.proficiency;
      return obj(p) && selection && ["mode_id", "version", "language_code", "scheme_id", "scheme_version", "level_id"].every(k => p[k] === selection[k as keyof LearningSelection])
        && hints(saved.items) && validAssessment(saved.assessment, saved.items.length) && saved.items.every(i => { const s = spans.get(i.sentence_id); return s && i.language_code === settings.source_language
          && i.span.start >= s.start && i.span.end <= s.end && source.slice(i.span.start, i.span.end).join("") === i.source_form; });
    });
  });
}
export function validSubmission(v: unknown): v is Submission {
  if (!obj(v) || !text(v.doc_name) || !text(v.command_id) || typeof v.replay !== "boolean"
    || !["claimed", "reuse", "coalesced", "reconcile", "retry_required", "no_content", "denied", "unavailable", "unknown", "missing", "superseded", "stale", "failed", "succeeded", "resuming"].includes(v.outcome)
    || (v.operation !== null && (!handle(v.operation) || v.operation.doc_name !== v.doc_name))) return false;
  if (["claimed", "coalesced", "reconcile", "retry_required", "resuming", "succeeded", "failed", "stale", "superseded"].includes(v.outcome) && !v.operation) return false;
  if (["reuse", "no_content", "denied", "missing"].includes(v.outcome) && v.operation !== null) return false;
  return (v.outcome === "denied" ? text(v.reason) : v.reason === null) && (["reuse", "succeeded"].includes(v.outcome) ? positive(v.result_revision) : v.result_revision === null);
}
export function validOperation(v: unknown): v is Operation {
  if (!obj(v) || !handle(v.handle) || ![...states, "missing"].includes(v.state) || ![v.source_current, v.settings_current, v.owns_current_slot, v.outcome_uncertain].every(x => typeof x === "boolean")
    || !nat(v.completed_batches) || !nat(v.total_batches) || v.completed_batches > v.total_batches || (v.failure_code !== null && !["provider_rejected", "invalid_output", "provider_outcome_unknown", "worker_stopped", "lease_expired", "binding_changed"].includes(v.failure_code))) return false;
  if (v.state === "missing") return [v.durable_state, v.component, v.content_block_id, v.source_token, v.result_revision, v.failure_code].every(x => x === null)
    && !v.source_current && !v.settings_current && !v.owns_current_slot && !v.outcome_uncertain && v.total_batches === 0;
  return states.includes(v.durable_state) && component(v.component) && text(v.content_block_id) && token(v.source_token) && positive(v.total_batches)
    && v.state === (v.source_current && v.settings_current ? v.durable_state : "stale")
    && (v.durable_state === "succeeded" ? positive(v.result_revision) && v.completed_batches === v.total_batches : v.result_revision === null)
    && (v.durable_state === "stale" || v.outcome_uncertain === ["running", "unknown"].includes(v.durable_state))
    && (v.durable_state !== "claimed" || v.completed_batches === 0) && (v.durable_state !== "partial" || v.completed_batches > 0)
    && (v.durable_state !== "running" || v.completed_batches < v.total_batches);
}
export class LearningService {
  constructor(private readonly client = new RestClient()) {}
  private async call<T>(path: string, body: object, signal: AbortSignal, valid: (v: unknown) => v is T, mutation = false): Promise<T> {
    try {
      const r = await this.client.requestJsonWithStatus<unknown>(`/documents/reading-interactions/learning/${path}`, { method: "POST", body: JSON.stringify(body), signal, cache: "no-store" });
      if (!r.ok) { const d = obj(r.payload) && obj(r.payload.detail) ? r.payload.detail : null;
        throw new LearningFailure(!mutation ? "read" : d && ["conflict", "invalid_request", "ineligible", "not_found", "unavailable"].includes(d.kind) ? "rejected" : "unknown", text(d?.code) ? d.code : ""); }
      if (!valid(r.payload)) throw new LearningFailure(mutation ? "unknown" : "read");
      return r.payload;
    } catch (e) { if (signal.aborted) throw e; throw e instanceof LearningFailure ? e : new LearningFailure(mutation ? "unknown" : "read"); }
  }
  read(doc: string, ids: string[], signal: AbortSignal) {
    if (!text(doc) || ids.length < 1 || ids.length > 50 || ids.some(id => !text(id)) || new Set(ids).size !== ids.length) return Promise.reject(new LearningFailure("read"));
    return this.call("read", { doc_name: doc, content_block_ids: ids }, signal, (v): v is LearningRead => validRead(v) && v.doc_name === doc
      && v.blocks.length === ids.length && v.blocks.every((b, i) => b.content_block_id === ids[i]));
  }
  generate(body: LearningCommand, signal: AbortSignal) { return this.call("generate", body, signal, (v): v is Submission => validSubmission(v) && v.doc_name === body.doc_name && v.command_id === body.command_id, true); }
  operation(h: LearningHandle, signal: AbortSignal) { return this.call("operations/read", { handle: h }, signal, (v): v is Operation => validOperation(v) && equalHandle(v.handle, h)); }
  wait(h: LearningHandle, signal: AbortSignal) { return this.call("operations/wait", { handle: h }, signal, (v): v is Operation => validOperation(v) && equalHandle(v.handle, h)); }
  reconcile(h: LearningHandle, signal: AbortSignal) { return this.call("operations/reconcile", { handle: h }, signal, (v): v is Submission => validSubmission(v) && v.doc_name === h.doc_name && v.command_id === h.operation_id && (!v.operation || equalHandle(v.operation, h)), true); }
  command(doc: string, id: string, signal: AbortSignal) { return this.call("commands/read", { doc_name: doc, command_id: id }, signal,
    (v): v is CommandReceipt => obj(v) && v.doc_name === doc && v.command_id === id && typeof v.found === "boolean"
      && (v.found ? validSubmission(v.receipt) && v.receipt.doc_name === doc && v.receipt.command_id === id : v.receipt === null)); }
  partial(h: LearningHandle, offset: number, signal: AbortSignal) { return this.call("operations/partial", { handle: h, offset, limit: 50 }, signal,
    (v): v is PartialRead => obj(v) && handle(v.handle) && equalHandle(v.handle, h) && component(v.component) && token(v.source_token) && v.offset === offset
      && nat(v.completed_batches) && positive(v.total_batches) && v.completed_batches <= v.total_batches && Array.isArray(v.batches) && v.batches.length <= 50
      && v.batches.every((b, i) => obj(b) && b.component === v.component && b.batch_index === offset + i && (b.component === "translation" ? rows(b.rows) : hints(b.items) && validAssessment(b.assessment, b.items.length)))
      && (!v.batches.length ? offset >= v.completed_batches : offset + v.batches.length <= v.completed_batches)
      && v.next_offset === (offset + v.batches.length < v.completed_batches ? offset + v.batches.length : null)); }
}
