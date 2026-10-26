import type { CleanupHistory, CleanupRecord, ReadingRefresh } from "../../services/ReadingContentService";

export const CLEANUP_PAGE_SIZE = 10;

export function mergeCleanupChange(history: CleanupHistory | undefined, refresh: ReadingRefresh,
  recordId: string, deleted: boolean): CleanupHistory {
  const change = refresh.cleanup_change;
  if (!history || !refresh.scope_current || refresh.mutation_revision !== refresh.revision
    || refresh.cleanup_record_id !== recordId || !change || change.record_id !== recordId
    || change.deleted !== deleted || (deleted && change.record != null)
    || (!deleted && (!change.record || change.record.record_id !== recordId))) {
    throw new Error("Cleanup history changed. Reload before continuing.");
  }
  const batch = history.items.find(item => item.revision === change.batch_revision);
  if (!batch || batch.deletions.filter(row => row.record_id === recordId).length !== 1) {
    throw new Error("Cleanup record is no longer loaded. Reload before continuing.");
  }
  const deletions = deleted ? batch.deletions.filter(row => row.record_id !== recordId)
    : batch.deletions.map(row => row.record_id === recordId ? change.record! : row);
  if (deletions.length !== change.applied_count || !Number.isInteger(change.remaining_count)
    || change.remaining_count < 0 || change.remaining_count > change.applied_count) {
    throw new Error("Cleanup history changed. Reload before continuing.");
  }
  return { ...history, items: history.items.map(item => item !== batch ? item : {
    ...batch, deletions, applied_count: change.applied_count, remaining_count: change.remaining_count,
  }) };
}

export function editableCleanupRecord(row: CleanupRecord) {
  return typeof row.record_id === "string" && /^[a-f0-9]{64}$/.test(row.record_id)
    && typeof row.text === "string" && (row.status === "cleaned" || row.status === "restored");
}

export function restoreUnavailable(row: CleanupRecord) {
  if (!editableCleanupRecord(row)) return "Record recovery is unavailable.";
  const reasons: Record<string, string> = {
    already_restored: "Already restored", block_deleted: "The block was deleted.",
    block_edited: "The block was manually edited.", source_changed: "The source changed.",
    mapping_changed: "The block mapping changed.", mapping_unavailable: "Content location is unavailable.",
  };
  return row.can_restore === true ? "" : reasons[row.restore_reason ?? ""] ?? "Restoration is unavailable.";
}
