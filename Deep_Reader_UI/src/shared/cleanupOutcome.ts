import type { CleanupOutcome } from "../types/api";

export function validCleanup(value: CleanupOutcome | null | undefined): value is CleanupOutcome {
  if (!value || !["not_applicable", "no_changes", "completed", "partial", "skipped", "failed"].includes(value.status)) return false;
  if (![value.applied_count, value.skipped_count, value.reading_revision].every(n => Number.isSafeInteger(n) && n >= 0)
      || (value.proposed_count !== null && (!Number.isSafeInteger(value.proposed_count) || value.proposed_count < 0))) return false;
  const changed = value.status === "completed" || value.status === "partial";
  return value.reading_content_changed === changed && (value.applied_count > 0) === changed
    && (!changed || value.reading_revision > 0)
    && (value.proposed_count === null ? value.applied_count + value.skipped_count === 0
      : value.proposed_count === value.applied_count + value.skipped_count);
}

export function cleanupMessage(cleanup?: CleanupOutcome | null) {
  if (!cleanup) return "";
  switch (cleanup.status) {
    case "completed": return "OCR cleanup completed. Original text is retained; cleanup can be undone.";
    case "partial": return "Some OCR fragments were cleaned. Uncertain text was kept; you can continue reading.";
    case "skipped": return "OCR cleanup was skipped. The document remains available for reading.";
    case "failed": return "OCR cleanup could not finish. The saved structure is available; no cleanup retry is required.";
    case "no_changes": return "No OCR cleanup changes were needed.";
    case "not_applicable": return "";
  }
}
