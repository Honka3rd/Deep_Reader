import type { BookPreferences } from "../../services/ReadingPreferencesService";
import type { LearningBlock, LearningCommand, LearningComponent, LearningHandle, LearningView, PartialRead } from "../../services/LearningService";
import type { ReaderContentBlock } from "../reader-content/model";

export interface PageSource { id: string; text: string }
export type Phase = "queued" | "reading" | "pending" | "ready" | "empty" | "partial" | "failed" | "unknown" | "stale" | "unavailable" | "error";
export interface LearningItem { blockId: string; component: LearningComponent; phase: Phase; busy?: boolean; awaitingCompletion?: boolean; view: LearningView; message: string; partial?: PartialRead }
export interface LearningPageState { loading: boolean; error: string; blocks: Record<string, LearningBlock>; items: Record<string, LearningItem> }
export const emptyPage = (): LearningPageState => ({ loading: true, error: "", blocks: {}, items: {} });
const activeOperations = new Set(["claimed", "running", "partial"]);
export function pageProgress(state: LearningPageState, expectedTasks: number) {
  let completed = 0, active = 0, attention = 0, checking = 0;
  for (const item of Object.values(state.items)) {
    const unconfirmed = item.awaitingCompletion ?? ((item.phase === "unknown" && !item.view.operation)
      || (item.phase === "error" && !!item.view.operation_state && activeOperations.has(item.view.operation_state)));
    const pending = ["queued", "reading", "pending", "partial"].includes(item.phase);
    if (item.busy || pending || unconfirmed) {
      active++;
      if (!item.busy && (unconfirmed || ["pending", "partial"].includes(item.phase))) checking++;
    } else if (item.phase === "ready" || item.phase === "empty") completed++;
    else attention++;
  }
  const total = Math.max(expectedTasks, Object.keys(state.items).length);
  return { completed, total, active, attention, checking,
    percent: total ? completed / total * 100 : 0,
    blocked: state.loading || active > 0 };
}
export const itemKey = (id: string, component: LearningComponent) => JSON.stringify([id, component]);
export function recoveryKey(book: BookPreferences, block: LearningBlock, component: LearningComponent) {
  return JSON.stringify([book.doc_name, book.revision, book.workspace.revision, block.content_block_id, block.source_token, component]);
}
export interface Recovery { command: LearningCommand; handle: LearningHandle | null; uncertain: boolean; blocked: boolean }
// Explicit reader-session owner in App, not storage or an accidental surviving callback.
// Only unresolved command identities survive page/controller teardown; no source/results retained.
export class LearningSessionMemory { readonly commands = new Map<string, Recovery>(); }
export function pageSources(doc: string, blocks: ReaderContentBlock[]): PageSource[] {
  // block_type is a presentation label (segmented prose is "paragraph").
  // The backend resolves learning sources by hierarchy-owned block ID, not type.
  if (blocks.some(b => !b.block_id?.trim() || !b.ownerContext.taskUnitId || b.ownerContext.documentId !== doc)
    || new Set(blocks.map(b => b.block_id)).size !== blocks.length) throw new Error("The current page identity could not be verified.");
  return blocks.map(b => ({ id: b.block_id, text: b.content }));
}
export function phaseFor(view: LearningView): Phase {
  if (view.operation_state && ["failed", "unknown", "stale", "partial"].includes(view.operation_state)) return view.operation_state as Phase;
  if (view.operation_state === "running" || view.operation_state === "claimed") return "pending";
  return view.state === "missing" ? "queued" : view.state;
}

export type ComponentAction = "refresh" | "retry" | "confirm" | "check";
export function componentAction(item: LearningItem): ComponentAction {
  if (item.phase === "ready" && item.view.saved) return "refresh";
  if (item.phase === "unknown" && item.view.operation) return "confirm";
  if (item.phase === "failed" || (item.phase === "unavailable" && item.view.state === "missing")) return "retry";
  return "check";
}
export function learningRows(item?: LearningItem) {
  if (item?.view.saved?.component === "translation") return { rows: item.view.saved.rows, partial: false };
  return { rows: item?.partial?.batches.flatMap(b => b.component === "translation" ? b.rows : []) ?? [], partial: !!item?.partial };
}
export function learningHints(item?: LearningItem) {
  if (item?.view.saved?.component === "hints") return { items: item.view.saved.items, partial: false, saved: true, assessment: item.view.saved.assessment };
  return { items: item?.partial?.batches.flatMap(b => b.component === "hints" ? b.items : []) ?? [], partial: !!item?.partial, saved: false, assessment: null };
}
