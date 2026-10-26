import type { EnhancementPreflightResponse, ReparseDocumentStructureResponse } from "../../types/api";
import { validCleanup } from "../../shared/cleanupOutcome";
export { validCleanup, cleanupMessage } from "../../shared/cleanupOutcome";

export type EnhancementChange = "unchanged" | "titles_only" | "structure_changed";

// Fail closed on old or contradictory responses, even when HTTP reports success.
export function confirmedChange(result: Partial<ReparseDocumentStructureResponse>, name: string): EnhancementChange | null {
  if (result.success !== true || result.doc_name !== name || result.parser_mode !== "llm_enhanced") return null;
  const kind = result.change_kind;
  if (kind !== "unchanged" && kind !== "titles_only" && kind !== "structure_changed") return null;
  if (result.committed !== (kind !== "unchanged")
      || result.interactions_cleared !== (kind === "structure_changed")) return null;
  if (result.phase != null || result.cleanup != null) {
    if (!validCleanup(result.cleanup) || !Number.isInteger(result.structure_version) || result.structure_version! < 1
        || !Number.isInteger(result.pending_review_count) || result.pending_review_count! < 0
        || result.review_required !== (result.pending_review_count! > 0)) return null;
    if (result.phase === "complete") {
      if (result.reading_layout_required !== false || result.retryable !== false || result.retry_stage != null) return null;
    } else if (result.phase === "layout") {
      if (result.reading_layout_required !== true || result.retryable !== true || result.retry_stage !== "layout") return null;
    } else if (result.phase === "superseded") {
      if (result.reading_layout_required != null || result.retryable !== false) return null;
    } else return null;
  } else if (result.reading_layout_required !== (kind === "structure_changed")) return null;
  return kind;
}

export function cleanupConsentValid(data: EnhancementPreflightResponse) {
  const cleanup = data.ocr_cleanup;
  return cleanup == null || (cleanup.enabled === true && cleanup.scope === "ocr_cleanup_reading_overlay_v1"
    && cleanup.policy_version === "ocr_cleanup_policy_v1"
    && cleanup.invalidation === "generated_interactions_only_if_reading_content_changed"
    && cleanup.preserves_notes_when_structure_unchanged === true
    && Number.isInteger(cleanup.reading_revision) && cleanup.reading_revision >= 0
    && ["eligible", "not_applicable", "skipped"].includes(cleanup.eligibility));
}
