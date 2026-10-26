import type { CleanupOperation } from "../../services/SmartCleanupService";
import { cleanupMessage } from "../../shared/cleanupOutcome";

export function unavailableMessage(reason: string | null) {
  switch (reason) {
    case "native_source": return "This document does not contain OCR text to clean.";
    case "ocr_evidence_missing": case "ocr_evidence_mismatch": case "extraction_mode_unsupported":
      return "The document's OCR source cannot be verified. Cleanup is unavailable.";
    case "capacity_exceeded": return "This document exceeds the current model's cleanup capacity. No cleanup was started.";
    case "capacity_unavailable": return "Cleanup capacity cannot be checked right now. No cleanup was started.";
    case "cleanup_server_unsupported": case "cleanup_response_invalid": return "This server does not support the required smart cleanup interface.";
    case "structure_not_ready": case "cleanup_source_not_ready": return "An accepted document is required before smart cleanup.";
    case "cleanup_document_missing": return "This document is no longer available.";
    case "cleanup_in_progress": return "Another document operation is in progress. No additional cleanup was started.";
    default: return "Smart cleanup is currently unavailable. No cleanup was started.";
  }
}
export function operationMessage(result: CleanupOperation) {
  if (result.phase === "superseded") return "This cleanup belongs to an earlier reading version. Current content was retained.";
  if (result.phase === "layout") return "Cleanup was saved, but reading is not ready yet. Resume reading without repeating cleanup.";
  if (result.phase !== "finished") return "The cleanup result is not confirmed. Do not start another cleanup.";
  if (["source_changed", "reading_revision_changed", "cleanup_policy_changed"].includes(result.reason ?? "")) {
    return "The document or cleanup requirements changed. No cleanup changes were saved. Check again before retrying.";
  }
  if (result.outcome?.status === "failed") return "Cleanup could not finish. No cleanup changes were saved.";
  if (result.outcome?.status === "skipped") return "Cleanup was skipped. No cleanup changes were saved.";
  if (result.outcome?.status === "not_applicable") return unavailableMessage(result.outcome.reason ?? null);
  return cleanupMessage(result.outcome);
}
