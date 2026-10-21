import { RestClient } from "./RestClient";
import type { ReadingInteractionTargetRequest, ReadingInteractionTargetResponse, ReadingInteractionResponseEnvelope } from "../types/api";
import { assertInteractionIdentity } from "../shared/readerIdentity";

export interface SummaryResponse {
  envelope: ReadingInteractionResponseEnvelope;
  payload: { headline: string; body: string; key_points: string[] } | null;
}
export interface NoteResponse {
  target: ReadingInteractionTargetResponse;
  target_token: string;
  max_chars: number;
  source_stale?: boolean;
  note: { note_id: string; text: string; revision: number; created_at: string; updated_at: string } | null;
}

export class SummaryNoteService {
  constructor(private client = new RestClient()) {}

  private validateTarget(request: ReadingInteractionTargetRequest, target: ReadingInteractionTargetResponse) {
    if (!target || target.doc_name !== request.doc_name || target.target_type !== request.target_type
      || (["chapter_id", "section_id", "task_unit_id", "content_block_id"] as const)
        .some(key => request[key] && request[key] !== target[key])) {
      throw new Error("Interaction target mismatch. Reopen the current target.");
    }
  }

  async summary(target: ReadingInteractionTargetRequest, operation: "read" | "generate" | "refresh", signal?: AbortSignal) {
    assertInteractionIdentity(target);
    const result = await this.client.requestJsonWithStatus<SummaryResponse>(
      `/documents/reading-interactions/summary/${operation}`,
      { method: "POST", body: JSON.stringify({ target }), signal });
    if (!result.payload?.envelope || result.payload.envelope.interaction_type !== "summary") {
      throw new Error("Summary could not be loaded. Please retry.");
    }
    this.validateTarget(target, result.payload.envelope.target);
    const payload = result.payload.payload;
    if (result.payload.envelope.status === "completed" && (!payload || typeof payload.headline !== "string"
      || typeof payload.body !== "string" || !Array.isArray(payload.key_points)
      || !payload.key_points.every(point => typeof point === "string"))) throw new Error("Invalid summary response.");
    return result.payload;
  }

  async note(target: ReadingInteractionTargetRequest, operation: "read" | "save" | "delete",
       fields: { text?: string; expected_revision?: number; mutation_id?: string; target_token?: string } = {},
       signal?: AbortSignal): Promise<NoteResponse> {
    assertInteractionIdentity(target);
    const result = await this.client.requestJson<NoteResponse>(`/documents/reading-interactions/note/${operation}`,
      { method: "POST", body: JSON.stringify({ target, ...fields }), signal });
    this.validateTarget(target, result.target);
    if (!result.target_token || result.max_chars !== 10000 || (result.note !== null
      && (!result.note || typeof result.note.text !== "string" || !Number.isInteger(result.note.revision)
        || result.note.revision < 1 || !result.note.note_id))) throw new Error("Invalid note response.");
    return result;
  }
}
