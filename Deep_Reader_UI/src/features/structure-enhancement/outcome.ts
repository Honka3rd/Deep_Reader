import type { ReparseDocumentStructureResponse } from "../../types/api";

export type EnhancementChange = "unchanged" | "titles_only" | "structure_changed";

// Fail closed on old or contradictory responses, even when HTTP reports success.
export function confirmedChange(result: Partial<ReparseDocumentStructureResponse>, name: string): EnhancementChange | null {
  if (result.success !== true || result.doc_name !== name || result.parser_mode !== "llm_enhanced") return null;
  const kind = result.change_kind;
  if (kind !== "unchanged" && kind !== "titles_only" && kind !== "structure_changed") return null;
  if (result.committed !== (kind !== "unchanged")
      || result.interactions_cleared !== (kind === "structure_changed")
      || result.reading_layout_required !== (kind === "structure_changed")) return null;
  return kind;
}
