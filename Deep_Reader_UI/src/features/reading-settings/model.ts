import type { BookPreferences, LearningSelection, ModeSelection, ProficiencyScheme, ReadingOptions } from "../../services/ReadingPreferencesService";

export interface SettingsDraft { mode: string; override: string | null; learning: LearningSelection | null }
export interface SettingsContext { book: BookPreferences; options: ReadingOptions }
export function initialDraft(book: BookPreferences, mode = book.active_mode): SettingsDraft {
  return { mode, override: book.source_language_override,
    learning: book.saved_settings.find(s => s.mode_id === "language_learning")?.selection ?? null };
}
export function sourceLanguage(draft: SettingsDraft, book: BookPreferences) {
  // An overridden response does not expose the hidden profile language.
  return draft.override ?? (book.source_origin === "profile" ? book.source_language : null);
}
export function schemeFor(draft: SettingsDraft, context: SettingsContext): ProficiencyScheme | undefined {
  return context.options.proficiency_schemes.find(s => s.language_code === sourceLanguage(draft, context.book));
}
export function languageLabel(code: string): string {
  try { return `${new Intl.DisplayNames(["en"], { type: "language" }).of(code) || code} (${code})`; }
  catch { return code; }
}
export function learningProblems(draft: SettingsDraft, { book, options }: SettingsContext): string[] {
  const problems: string[] = [];
  const source = sourceLanguage(draft, book), native = book.workspace.native_language;
  if (!native || !options.language_codes.includes(native)) problems.push("Set a supported native language in Workspace preferences.");
  if (!source || !options.language_codes.includes(source)) problems.push(draft.override === null && book.source_origin === "override"
    ? "To restore the saved profile, first save Normal reading with Use saved profile, then reopen Language learning."
    : "Choose the book’s original language. The saved profile has no supported language.");
  if (source && source === native) problems.push("Language learning is unavailable because the original language matches your native language.");
  const scheme = schemeFor(draft, { book, options }), value = draft.learning;
  if (!scheme || !value || value.language_code !== source || value.scheme_id !== scheme.scheme_id
    || value.scheme_version !== scheme.version || !scheme.levels.some(l => l.level_id === value.level_id)) problems.push("Choose your reading level for this book’s original language.");
  return problems;
}
export function selectedLevel(draft: SettingsDraft, context: SettingsContext) {
  const scheme = schemeFor(draft, context), value = draft.learning;
  return scheme && value?.language_code === scheme.language_code && value.scheme_id === scheme.scheme_id && value.scheme_version === scheme.version
    && scheme.levels.some(l => l.level_id === value.level_id) ? value.level_id : "";
}
export function withLevel(draft: SettingsDraft, context: SettingsContext, level: string): SettingsDraft {
  const scheme = schemeFor(draft, context);
  return { ...draft, learning: scheme && scheme.levels.some(l => l.level_id === level) ? {
    mode_id: "language_learning", version: "1", language_code: scheme.language_code,
    scheme_id: scheme.scheme_id, scheme_version: scheme.version, level_id: level,
  } : null };
}
function sameLearning(left: LearningSelection | null, right: LearningSelection | null) {
  if (!left || !right) return left === right;
  return left.mode_id === right.mode_id && left.version === right.version && left.language_code === right.language_code
    && left.scheme_id === right.scheme_id && left.scheme_version === right.scheme_version && left.level_id === right.level_id;
}
export function draftChanged(draft: SettingsDraft, book: BookPreferences) {
  const initial = initialDraft(book);
  return draft.mode !== initial.mode || draft.override !== initial.override
    || (draft.mode === "language_learning" && !sameLearning(draft.learning, initial.learning));
}
export interface SettingsFeaturePolicy {
  version: string;
  problems(draft: SettingsDraft, context: SettingsContext): string[];
  selection(draft: SettingsDraft): ModeSelection;
}

// These server reasons can be resolved by the form fields before applying. Other
// Unknown policy remains blocking. The API has verified canonical book existence;
// explicit CAS apply can reactivate preferences after same-name reimport.
const editableReasons = new Set(["profile_missing", "profile_invalid", "profile_unavailable", "source_language_required", "source_language_invalid",
  "native_language_required", "native_language_invalid", "same_language", "proficiency_required", "proficiency_language_mismatch", "proficiency_invalid", "settings_unsupported"]);
export function serverBlockers(book: BookPreferences, mode: string): string[] {
  const reasons = book.modes.find(m => m.mode_id === mode)?.reasons ?? [];
  return reasons.filter(r => r.code !== "book_deleted" && (mode !== "language_learning" || !editableReasons.has(r.code))).map(r => r.message);
}
