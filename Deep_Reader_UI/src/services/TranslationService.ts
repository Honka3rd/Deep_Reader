import { RestClient } from "./RestClient";
import type { ReadingInteractionTargetResponse } from "../types/api";

export interface TranslationTarget { doc_name: string; content_block_id: string }
export interface TranslationSavedRecord {
  source_language: string;
  target_language: string;
  status: "completed" | "partial";
  translated_text: string;
  generated_at: string;
  updated_at: string;
  revision: number;
  mutation_id: string;
  effective_content_hash: string;
}
export interface TranslationReadResponse {
  target: ReadingInteractionTargetResponse;
  language_codes: string[];
  default_source_language: string | null;
  effective_content_hash: string;
  reading_revision: number;
  translations: TranslationSavedRecord[];
}
export interface TranslationGenerateRequest extends TranslationTarget {
  target_language: string;
  source_language?: string;
  mutation_id?: string;
}
export interface TranslationGenerateResponse {
  envelope: {
    target: ReadingInteractionTargetResponse;
    status: "completed" | "partial" | "untranslatable";
    reading_revision: number;
    effective_content_hash: string;
    persisted: boolean;
  };
  payload: { source_language: string; target_language: string; translated_text: string | null };
  record: TranslationSavedRecord | null;
}

const invalid = () => new Error("Translation response could not be verified. Reload saved translations.");
const nonempty = (value: unknown): value is string => typeof value === "string" && Boolean(value.trim());
const revision = (value: unknown) => Number.isInteger(value) && Number(value) >= 0;
export class TranslationRequestRejected extends Error {
  override name = "TranslationRequestRejected";
}
function assertTarget(request: TranslationTarget, target: ReadingInteractionTargetResponse) {
  if (!target || target.doc_name !== request.doc_name || target.target_type !== "content_block"
    || target.content_block_id !== request.content_block_id || target.target_id !== request.content_block_id) throw invalid();
}
function assertIdentity(target: TranslationTarget) {
  if (!nonempty(target.doc_name) || !nonempty(target.content_block_id)) throw new Error("Select a content block first.");
}
function assertRecord(record: TranslationSavedRecord) {
  if (!record || !nonempty(record.source_language) || !nonempty(record.target_language)
    || !["completed", "partial"].includes(record.status) || !nonempty(record.translated_text)
    || !nonempty(record.mutation_id) || !nonempty(record.effective_content_hash)
    || !revision(record.revision) || record.revision < 1
    || !nonempty(record.generated_at) || !Number.isFinite(Date.parse(record.generated_at))
    || !nonempty(record.updated_at) || !Number.isFinite(Date.parse(record.updated_at))) throw invalid();
}

export class TranslationService {
  constructor(private readonly client = new RestClient()) {}

  async read(target: TranslationTarget, signal?: AbortSignal): Promise<TranslationReadResponse> {
    assertIdentity(target);
    const result = await this.client.requestJsonWithStatus<TranslationReadResponse>(
      "/documents/reading-interactions/translate/read", { method: "POST", signal, cache: "no-store",
        body: JSON.stringify({ doc_name: target.doc_name, content_block_id: target.content_block_id }) });
    if (!result.ok) throw new Error("Saved translations could not be loaded. Please retry.");
    const data = result.payload;
    if (!data) throw invalid();
    assertTarget(target, data.target);
    if (!Array.isArray(data.language_codes) || !data.language_codes.every(nonempty)
      || new Set(data.language_codes).size !== data.language_codes.length
      || !(data.default_source_language === null || typeof data.default_source_language === "string")
      || !nonempty(data.effective_content_hash) || !revision(data.reading_revision)
      || !Array.isArray(data.translations)) throw invalid();
    const codes = new Set<string>();
    for (const record of data.translations) {
      assertRecord(record);
      if (codes.has(record.target_language) || record.effective_content_hash !== data.effective_content_hash) throw invalid();
      codes.add(record.target_language);
    }
    return data;
  }

  async generate(request: TranslationGenerateRequest, signal?: AbortSignal): Promise<TranslationGenerateResponse> {
    assertIdentity(request);
    if (!nonempty(request.target_language) || (request.source_language !== undefined && !nonempty(request.source_language))) {
      throw new Error("Select languages from the available options.");
    }
    if (request.source_language === request.target_language) {
      throw new TranslationRequestRejected("Choose a target language different from the source language.");
    }
    const result = await this.client.requestJsonWithStatus<TranslationGenerateResponse>(
      "/documents/reading-interactions/translate/generate", { method: "POST", signal,
        body: JSON.stringify({ doc_name: request.doc_name, content_block_id: request.content_block_id,
          target_language: request.target_language, ...(request.source_language ? { source_language: request.source_language } : {}),
          ...(request.mutation_id ? { mutation_id: request.mutation_id } : {}) }) });
    if (!result.ok) {
      const detail = (result.payload as unknown as { detail?: unknown } | null)?.detail;
      const providerRejected = result.status === 502
        && (detail === "translation_provider_failed" || detail === "translation_invalid_output");
      const preflightRejected = result.status === 503
        && (detail === "translation_capacity_unavailable" || detail === "translation_profile_unavailable");
      const message = detail === "translation_same_language" ? "Choose a target language different from the source language."
        : result.status === 409 ? "The source or saved translation changed. Reload before trying again."
        : providerRejected ? "The translation service failed to return a usable result. Please try again."
        : preflightRejected ? "Translation is temporarily unavailable. Please try again."
        : "Translation could not be confirmed. Check saved translations before retrying.";
      if ((result.status >= 400 && result.status < 500 && result.status !== 408) || providerRejected || preflightRejected) {
        throw new TranslationRequestRejected(message);
      }
      throw new Error(message);
    }
    const data = result.payload;
    if (!data?.envelope || !data.payload) throw invalid();
    assertTarget(request, data.envelope.target);
    const { envelope, payload, record } = data;
    if (!["completed", "partial", "untranslatable"].includes(envelope.status)
      || typeof envelope.persisted !== "boolean" || !revision(envelope.reading_revision)
      || !nonempty(envelope.effective_content_hash) || !nonempty(payload.source_language)
      || payload.target_language !== request.target_language
      || (envelope.status === "untranslatable" ? payload.translated_text !== null : !nonempty(payload.translated_text))
      || (!envelope.persisted && request.source_language && payload.source_language !== request.source_language)) throw invalid();
    if (envelope.persisted) {
      assertRecord(record!);
      if (record!.status !== envelope.status || record!.target_language !== payload.target_language
        || record!.source_language !== payload.source_language || record!.translated_text !== payload.translated_text
        || record!.effective_content_hash !== envelope.effective_content_hash) throw invalid();
    } else if (record !== null || envelope.status !== "untranslatable") throw invalid();
    return data;
  }
}
export const translationService = new TranslationService();
