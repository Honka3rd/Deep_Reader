import type { ReadingInteractionResponseEnvelope, ReadingInteractionTargetRequest } from "../../types/api";

type InteractionResponse = { envelope: ReadingInteractionResponseEnvelope; payload?: unknown };

/** A user-selected action, never a passive read or navigation effect. */
export async function readOrGenerate<T extends InteractionResponse>({
  target, interactionType, read, generate, isCurrent, onGenerating, retryGeneration = false,
}: {
  retryGeneration?: boolean;
  target: ReadingInteractionTargetRequest;
  interactionType: string;
  read: () => Promise<T>;
  generate: () => Promise<T>;
  isCurrent: () => boolean;
  onGenerating: () => void;
}): Promise<T | undefined> {
  const validate = (response: T) => {
    const envelope = response?.envelope;
    const actual = envelope?.target;
    if (!actual || envelope.interaction_type !== interactionType
      || actual.doc_name !== target.doc_name || actual.target_type !== target.target_type
      || (["chapter_id", "section_id", "task_unit_id", "content_block_id"] as const)
        .some(key => target[key] && target[key] !== actual[key])
      || typeof envelope.status !== "string"
      || (envelope.status === "not_generated" && (response.payload !== null || envelope.artifact_id != null || envelope.session_id != null))) {
      throw new Error("Invalid interaction response. Reopen the current target.");
    }
    return response;
  };
  const saved = await read();
  if (!isCurrent()) return;
  validate(saved);
  const retryableFailure = ["generation_failed", "validation_failed", "insufficient_content"].includes(saved.envelope.status)
    && saved.payload == null;
  const outdatedQuestion = interactionType === "critical_thinking_session"
    && saved.envelope.status === "stale_target" && saved.envelope.reason === "interaction_policy_stale";
  if (saved.envelope.status !== "not_generated" && !(retryGeneration && (retryableFailure || outdatedQuestion))) return saved;
  onGenerating();
  if (!isCurrent()) return;
  const generated = await generate();
  if (!isCurrent()) return;
  return validate(generated);
}
