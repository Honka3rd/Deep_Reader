import type { TranslationReadResponse, TranslationSavedRecord, TranslationTarget } from "../../services/TranslationService";
import type { ReaderContentBlock } from "../reader-content/model";

export type TranslationSurface = "languages" | "saved" | null;
export type TranslationStatus = "idle" | "loading" | "ready" | "generating" | "failed";
export interface TranslationSelection extends TranslationTarget { sourceIdentity: string }
export function blockSourceIdentity(block: ReaderContentBlock) {
  return JSON.stringify([block.block_id, block.content, block.ownerContext?.sourceHash]);
}
export function translationKey(target: TranslationTarget) { return JSON.stringify([target.doc_name, target.content_block_id]); }
export function translationCommandKey(target: TranslationTarget, from: string, to: string) {
  return JSON.stringify([target.doc_name, target.content_block_id, from, to]);
}
export function profileFrom(data: TranslationReadResponse): string | null {
  return data.default_source_language && data.language_codes.includes(data.default_source_language)
    ? data.default_source_language : null;
}
export function validLanguages(codes: string[], from: string | null, to: string | null) {
  return Boolean(from && to && from !== to && codes.includes(from) && codes.includes(to));
}
export function targetLanguages(codes: string[], from: string | null) {
  return codes.filter(code => code !== from);
}
export function savedTranslation(data: TranslationReadResponse | null, to: string | null) {
  return to && data?.language_codes.includes(to)
    ? data.translations.find(record => record.target_language === to
      && record.effective_content_hash === data.effective_content_hash) : undefined;
}
export function languageLabel(code: string) {
  try {
    const name = new Intl.DisplayNames(["en"], { type: "language" }).of(code);
    return name && name !== code ? `${name} (${code})` : code;
  } catch { return code; }
}
export function savedSections(records: TranslationSavedRecord[], focusLanguage: string | null = null) {
  return [...records].sort((a, b) =>
    Number(b.target_language === focusLanguage) - Number(a.target_language === focusLanguage)
    || a.target_language.localeCompare(b.target_language));
}
export function replaceSaved(records: TranslationSavedRecord[], next: TranslationSavedRecord) {
  return savedSections([...records.filter(record => record.target_language !== next.target_language), next]);
}
