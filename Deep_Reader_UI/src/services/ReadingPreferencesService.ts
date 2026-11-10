import { RestClient } from "./RestClient";

export interface PreferenceIssue { code: string; field: string; message: string }
export interface ProficiencyScheme {
  language_code: string; scheme_id: string; version: string; label: string; scope_note: string;
  levels: { level_id: string; order: number; label: string; reading_description: string }[];
}
export interface ReadingOptions { language_codes: string[]; proficiency_schemes: ProficiencyScheme[] }
export interface WorkspacePreferences { revision: number; native_language: string | null }
export interface LearningSelection {
  mode_id: "language_learning"; version: "1"; language_code: string;
  scheme_id: string; scheme_version: string; level_id: string;
}
export type ModeSelection = { mode_id: "normal"; version: "1" } | LearningSelection;
export interface ReadingMode {
  mode_id: string; label: string; description: string; settings_version: string;
  eligible: boolean; reasons: PreferenceIssue[];
}
export interface BookPreferences {
  doc_name: string; revision: number; workspace: WorkspacePreferences;
  active_mode: string; effective_mode: "normal" | "language_learning" | null;
  source_language_override: string | null; source_language: string | null; source_origin: "profile" | "override";
  active_reasons: PreferenceIssue[]; modes: ReadingMode[];
  saved_settings: { mode_id: string; version: string; supported: boolean; selection: LearningSelection | null }[];
}
export interface BookApply {
  doc_name: string; expected_revision: number; expected_workspace_revision: number; mutation_id: string;
  selection: ModeSelection; source_language_override?: string | null;
}
export class PreferencesFailure extends Error {
  constructor(readonly kind: "read" | "rejected" | "conflict" | "unknown", readonly issues: PreferenceIssue[] = []) {
    super(kind === "unknown" ? "The save outcome is unknown. Read current settings before trying again."
      : kind === "conflict" ? "Settings changed elsewhere. Read current settings before applying your draft."
      : kind === "rejected" ? "Settings could not be applied. Check the selected options."
      : "Reading settings are unavailable. Try reading them again.");
  }
}
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const nullable = (v: unknown) => v === null || text(v);
const revision = (v: unknown) => Number.isSafeInteger(v) && Number(v) >= 0;
const issues = (v: unknown): v is PreferenceIssue[] => Array.isArray(v) && v.every(i => record(i) && text(i.code) && text(i.field) && text(i.message));
const selection = (v: unknown): v is LearningSelection => record(v) && v.mode_id === "language_learning" && v.version === "1"
  && [v.language_code, v.scheme_id, v.scheme_version, v.level_id].every(text);
export function isWorkspace(v: unknown): v is WorkspacePreferences {
  return record(v) && revision(v.revision) && nullable(v.native_language);
}
export function isBook(v: unknown): v is BookPreferences {
  if (!record(v) || !text(v.doc_name) || !revision(v.revision) || !isWorkspace(v.workspace) || !text(v.active_mode)
    || ![null, "normal", "language_learning"].includes(v.effective_mode as string | null)
    || !nullable(v.source_language_override) || !nullable(v.source_language) || !["profile", "override"].includes(v.source_origin as string)
    || !issues(v.active_reasons) || !Array.isArray(v.modes) || !Array.isArray(v.saved_settings)) return false;
  if ((v.effective_mode === null) !== (v.active_reasons.length > 0) || (v.effective_mode !== null && v.active_mode !== v.effective_mode)) return false;
  if (!v.modes.every(m => record(m) && [m.mode_id, m.label, m.description, m.settings_version].every(text)
    && typeof m.eligible === "boolean" && issues(m.reasons) && m.eligible === (m.reasons.length === 0))) return false;
  if (!v.saved_settings.every(m => record(m) && text(m.mode_id) && text(m.version) && typeof m.supported === "boolean"
    && (m.selection === null ? !m.supported : m.supported && selection(m.selection) && m.selection.mode_id === m.mode_id && m.selection.version === m.version))) return false;
  if (new Set(v.modes.map(m => m.mode_id)).size !== v.modes.length || new Set(v.saved_settings.map(m => m.mode_id)).size !== v.saved_settings.length) return false;
  const active = v.modes.find(m => m.mode_id === v.active_mode);
  if (active && JSON.stringify(active.reasons) !== JSON.stringify(v.active_reasons)) return false;
  if (v.effective_mode === "language_learning") {
    const saved = v.saved_settings.find(m => m.mode_id === "language_learning")?.selection;
    if (!saved || !v.source_language || !v.workspace.native_language || v.source_language === v.workspace.native_language || saved.language_code !== v.source_language) return false;
  }
  return true;
}
export function isOptions(v: unknown): v is ReadingOptions {
  return record(v) && Array.isArray(v.language_codes) && v.language_codes.length > 0 && v.language_codes.every(text)
    && new Set(v.language_codes).size === v.language_codes.length && Array.isArray(v.proficiency_schemes)
    && v.proficiency_schemes.length === v.language_codes.length
    && new Set(v.proficiency_schemes.map(s => record(s) ? s.language_code : null)).size === v.language_codes.length
    && v.proficiency_schemes.every(s => record(s) && (v.language_codes as string[]).includes(s.language_code as string)
      && [s.scheme_id, s.version, s.label, s.scope_note].every(text) && Array.isArray(s.levels) && s.levels.length > 0
      && new Set(s.levels.map(l => record(l) ? l.level_id : null)).size === s.levels.length
      && s.levels.every(l => record(l) && [l.level_id, l.label, l.reading_description].every(text) && revision(l.order) && Number(l.order) > 0));
}

export class ReadingPreferencesService {
  constructor(private readonly client = new RestClient()) {}
  private async request<T>(path: string, signal: AbortSignal, validate: (v: unknown) => v is T, body?: object, mutation = false, method = "POST"): Promise<T> {
    try {
      const response = await this.client.requestJsonWithStatus<unknown>(path, {
        method: body ? method : "GET", signal, cache: "no-store", ...(body ? { body: JSON.stringify(body) } : {}),
      });
      if (!response.ok) {
        const detail = record(response.payload) && record(response.payload.detail) ? response.payload.detail : null;
        const kind = !mutation ? "read" : detail?.kind === "conflict" ? "conflict"
          : ["invalid_request", "ineligible", "not_found", "unavailable"].includes(String(detail?.kind)) ? "rejected" : "unknown";
        throw new PreferencesFailure(kind, detail && issues(detail.reasons) ? detail.reasons : []);
      }
      if (!validate(response.payload)) throw new PreferencesFailure(mutation ? "unknown" : "read");
      return response.payload;
    } catch (error) {
      if (signal.aborted) throw error;
      throw error instanceof PreferencesFailure ? error : new PreferencesFailure(mutation ? "unknown" : "read");
    }
  }
  options(signal: AbortSignal) { return this.request("/reading-preferences/options", signal, isOptions); }
  workspace(signal: AbortSignal) { return this.request("/reading-preferences/workspace", signal, isWorkspace); }
  saveWorkspace(native_language: string | null, expected_revision: number, signal: AbortSignal) {
    return this.request("/reading-preferences/workspace", signal, isWorkspace,
      { native_language, expected_revision, mutation_id: crypto.randomUUID() }, true, "PUT");
  }
  book(doc: string, signal: AbortSignal) {
    if (!doc.trim()) return Promise.reject(new PreferencesFailure("read"));
    return this.request("/documents/reading-preferences/read", signal, (v): v is BookPreferences => isBook(v) && v.doc_name === doc, { doc_name: doc });
  }
  apply(body: Omit<BookApply, "mutation_id">, signal: AbortSignal) {
    return this.request("/documents/reading-preferences/apply", signal, (v): v is BookPreferences => isBook(v) && v.doc_name === body.doc_name,
      { ...body, mutation_id: crypto.randomUUID() }, true);
  }
}
export const readingPreferencesService = new ReadingPreferencesService();
