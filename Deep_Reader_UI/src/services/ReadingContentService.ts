import { RestClient } from "./RestClient";
import type { ContentBlock } from "../types/api";

export type ReadingRefreshBlock = { block_id: string; chapter_id: string; section_id: string; task_unit_id: string;
  block_order: number; insert_before_block_id?: string | null; unit_source_hash: string;
  deleted: boolean; manually_edited: boolean; content_block?: ContentBlock | null };
export type ReadingRefresh = { document_id: string; structure_version?: string | null; revision: number;
  source_token: string; mutation_id: string; mutation_revision: number; cleanup_record_id?: string | null;
  scope_current: boolean; generated_invalidated: boolean; cleared_block_id?: string | null;
  body_changed?: boolean; summaries_invalidated?: boolean; translation_block_ids?: string[];
  cleared_block_ids?: string[];
  cleanup_change?: CleanupRecordChange | null;
  blocks: ReadingRefreshBlock[]; block_page: { total: number; offset: number; limit: number } };

export type CleanupLink = { block_id: string; chapter_id: string; section_id: string; task_unit_id: string;
  char_from: number; char_to: number; can_navigate: boolean };
export type CleanupRecord = { quote: string; reason: string; page_indices: number[];
  record_id?: string; text?: string; edited_at?: string | null; links?: CleanupLink[];
  status?: "cleaned" | "restored"; can_restore?: boolean; restore_reason?: string | null; can_delete?: boolean };
export type CleanupRecordChange = { batch_revision: number; record_id: string; deleted: boolean;
  record?: CleanupRecord | null; applied_count: number; remaining_count: number };
export interface CleanupHistory {
  total: number; offset: number; limit: number;
  items: { revision: number; at: string; applied_count: number; skipped_count: number;
    remaining_count: number; undoable: boolean;
    deletions: CleanupRecord[] }[];
}

export type ReviewRegion = { id: string; page_index: number; reason?: string; raw_candidate?: string; status: string;
  repair?: { text: string; edited_at?: string | null; applied_text?: string | null; applied_at?: string | null;
    can_edit: boolean; can_apply: boolean; can_delete: boolean; blocked_reason?: string | null } | null;
  block_ids?: string[]; mapping_status?: "complete" | "partial" | "unavailable";
  mapping_reason?: string; body_status?: "linked" | "excluded" | "unknown" | "pending";
  related_deleted_count?: number;
  manual_change?: { status: "modified" | "deleted" | "restored"; at: string } | null };
export type ReadingBlock = { block_id: string; content: string; deleted: boolean; modified?: boolean; manually_edited?: boolean;
  review_repair?: { can_edit: boolean; can_delete: boolean; can_keep: boolean; blocked_reason?: string | null } | null;
  chapter_id?: string; section_id?: string; task_unit_id?: string };
export type ReviewNavigationBlock = ReadingBlock & { chapter_id: string; section_id: string; task_unit_id: string };
export type ReadingState = { document_id?: string; revision: number; source_token: string; disposition: string; can_undo: boolean;
  refresh?: ReadingRefresh | null;
  structure_version?: string | null;
  undo_action?: string | null; cleanup_audit?: CleanupHistory; block_page?: { total: number; offset: number; limit: number };
  max_chars: number; regions: ReviewRegion[]; blocks: ReadingBlock[] };
export type RegionRepairAction = "region_edit" | "region_apply" | "region_delete";
export type RegionRepairCommand = { region_id: string; text?: string; discard_interactions?: boolean };
export type ReviewBlockAction = "review_block_edit" | "review_block_delete" | "review_block_keep";
export type ReviewBlockCommand = { region_id: string; block_id: string; text?: string; discard_interactions?: boolean };
export type ReadingMutation = { action: "edit" | "delete" | "undo" | "discard_region" | "cleanup_edit" | "cleanup_restore" | "cleanup_delete" | RegionRepairAction | ReviewBlockAction;
  expected_revision: number; source_token: string; mutation_id: string; block_id?: string;
  region_id?: string; text?: string; discard_interactions?: boolean; cleanup_record_id?: string };

export class ReadingMutationRejected extends Error {
  constructor(message: string, public readonly status: number) { super(message); }
}

class ReadingContentService {
  private client = new RestClient();
  private path(name: string) {
    if (!name.trim()) throw new Error("Document is required.");
    return `/documents/${encodeURIComponent(name)}/reading-content`;
  }
  async read(name: string, query: URLSearchParams, signal: AbortSignal) {
    const state = await this.client.requestJson<ReadingState>(`${this.path(name)}?${query}`, { signal, cache: "no-store" });
    return { ...state, structure_version: state.structure_version ?? null,
      regions: state.regions.map(region => ({ ...region, manual_change: region.manual_change ?? null })) };
  }
  async mutate(name: string, command: ReadingMutation, signal: AbortSignal) {
    const result = await this.client.requestJsonWithStatus<ReadingState & { detail?: unknown }>(this.path(name),
      { method: "POST", body: JSON.stringify(command), signal });
    if (!result.ok) throw new ReadingMutationRejected(typeof result.payload?.detail === "string"
      ? result.payload.detail : `HTTP ${result.status}`, result.status);
    return result.payload;
  }
  readRefresh(name: string, binding: ReadingRefresh, offset: number, signal: AbortSignal) {
    const query = new URLSearchParams({ refresh_mutation_id: binding.mutation_id,
      expected_revision: String(binding.revision), source_token: binding.source_token, block_offset: String(offset) });
    return this.client.requestJson<ReadingRefresh>(`${this.path(name)}?${query}`, { signal, cache: "no-store" });
  }
}
export const readingContentService = new ReadingContentService();
