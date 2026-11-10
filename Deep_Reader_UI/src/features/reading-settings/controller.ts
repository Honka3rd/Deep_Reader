import { useEffect, useRef, useState } from "react";
import { PreferencesFailure, readingPreferencesService as service, type BookPreferences, type ReadingOptions, type WorkspacePreferences } from "../../services/ReadingPreferencesService";
import type { AppNotificationState } from "../../shared/components/AppNotification";
import { draftChanged, initialDraft, serverBlockers, type SettingsDraft } from "./model";
import { settingsFeatures } from "./features";

type BookTarget = { kind: "book"; doc: string; title: string };
type Target = BookTarget | { kind: "workspace"; back?: BookTarget };
const quiet: AppNotificationState = { open: false, message: "", severity: "info" };
export function useReadingSettingsController(owner: boolean | string, { closeOnBookSave = false } = {}) {
  const [target, setTarget] = useState<Target | null>(null);
  const [options, setOptions] = useState<ReadingOptions | null>(null);
  const [book, setBook] = useState<BookPreferences | null>(null);
  const [workspace, setWorkspace] = useState<WorkspacePreferences | null>(null);
  const [draft, setDraft] = useState<SettingsDraft | null>(null);
  const [native, setNative] = useState<string | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [busy, setBusy] = useState(false);
  const [recovery, setRecovery] = useState(false);
  const [detail, setDetail] = useState(false);
  const [notification, setNotification] = useState(quiet);
  const [confirm, setConfirm] = useState(false);
  const [returnFocus, setReturnFocus] = useState<string | null>(null);
  const request = useRef<{ id: number; abort: AbortController } | null>(null);
  const generation = useRef(0), pending = useRef(false), destination = useRef<(() => void) | null>(null);
  const targetRef = useRef(target); targetRef.current = target;
  const ownerRef = useRef(owner); ownerRef.current = owner;
  const dirty = target?.kind === "book" ? !!(draft && book && draftChanged(draft, book)) : !!workspace && native !== workspace.native_language;
  const notify = (message: string, severity: AppNotificationState["severity"] = "error") => setNotification({ open: true, message, severity });
  function cancel() { generation.current++; request.current?.abort.abort(); request.current = null; pending.current = false; }
  function begin() {
    cancel(); const targetOwner = targetRef.current, sessionOwner = ownerRef.current;
    const current = { id: generation.current, abort: new AbortController() }; request.current = current;
    return { signal: current.abort.signal, current: () => !current.abort.signal.aborted && current.id === generation.current
      && targetRef.current === targetOwner && ownerRef.current === sessionOwner };
  }
  useEffect(() => {
    cancel(); setTarget(null); setConfirm(false); setNotification(quiet); destination.current = null;
  }, [owner]);
  useEffect(() => {
    setBook(null); setWorkspace(null); setDraft(null); setNative(null); setOptions(null);
    setStatus("loading"); setBusy(false); setRecovery(false); setDetail(false); setConfirm(false); setNotification(quiet);
    destination.current = null;
    if (!target || !owner) return;
    if (typeof owner === "string" && (target.kind === "book" ? target.doc : target.back?.doc) !== owner) return;
    const work = begin();
    void Promise.all([service.options(work.signal), target.kind === "book" ? service.book(target.doc, work.signal) : service.workspace(work.signal)])
      .then(([catalog, value]) => {
        if (!work.current()) return;
        setOptions(catalog);
        if ("doc_name" in value) { setBook(value); setWorkspace(value.workspace); setDraft(initialDraft(value)); }
        else { setWorkspace(value); setNative(value.native_language); }
        setStatus("ready");
      }).catch(error => {
        if (!work.current()) return;
        setStatus("error"); notify(error instanceof PreferencesFailure ? error.message : "Reading settings are unavailable.");
      });
    return cancel;
  }, [target, owner]);
  useEffect(() => {
    if (!target || (!dirty && !busy && !recovery)) return;
    const protect = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", protect);
    return () => window.removeEventListener("beforeunload", protect);
  }, [target, dirty, busy, recovery]);
  function guard(action: () => void) {
    if (dirty || busy || recovery) { destination.current = action; setConfirm(true); setNotification(quiet); }
    else action();
  }
  function leave() { cancel(); setTarget(null); setNotification(quiet); }
  function close() { guard(() => { if (target?.kind === "workspace" && target.back) setTarget(target.back); else leave(); }); }
  function openBook(doc: string, title: string) {
    if (!doc.trim()) return;
    guard(() => { setReturnFocus(doc); setTarget({ kind: "book", doc, title }); });
  }
  function openWorkspace() {
    guard(() => { setReturnFocus(target?.kind === "book" ? target.doc : "workspace"); setTarget({ kind: "workspace", ...(target?.kind === "book" ? { back: target } : {}) }); });
  }
  function choose(mode: string) {
    if (!book || !draft || mode === draft.mode) { setDetail(true); return; }
    guard(() => { setDraft(initialDraft(book, mode)); setDetail(true); setNotification(quiet); });
  }
  function backToList() { guard(() => { if (book) setDraft(initialDraft(book)); setDetail(false); }); }
  async function reload() {
    if (!target || pending.current) return;
    if (status === "error") { setTarget({ ...target }); return; }
    const work = begin(); pending.current = true; setBusy(true); setNotification(quiet);
    try {
      const value = target.kind === "book" ? await service.book(target.doc, work.signal) : await service.workspace(work.signal);
      if (!work.current()) return;
      if ("doc_name" in value) { setBook(value); setWorkspace(value.workspace); }
      else { setWorkspace(value); }
      setRecovery(false); notify("Current settings loaded. Your draft is retained; review it before saving.", "info");
    } catch (error) { if (work.current()) notify(error instanceof PreferencesFailure ? error.message : "Could not read current settings."); }
    finally { if (work.current()) { pending.current = false; setBusy(false); } }
  }
  const mode = book?.modes.find(m => m.mode_id === draft?.mode);
  const feature = mode && settingsFeatures[mode.mode_id]?.version === mode.settings_version ? settingsFeatures[mode.mode_id] : undefined;
  const problems = draft && book && options ? feature ? [...feature.problems(draft, { book, options }), ...serverBlockers(book, draft.mode),
    ...(draft.override && !options.language_codes.includes(draft.override) ? ["Choose a supported original language."] : [])] : ["This mode is not supported by this client. Saved settings are preserved."] : [];
  // Normal reading has no learning prerequisites; backend still validates source overrides and lifecycle.
  const invalidNative = !!native && !options?.language_codes.includes(native);
  const canSave = status === "ready" && !busy && !recovery && (target?.kind === "workspace" ? !invalidNative && dirty : !!feature && !problems.length);
  async function save() {
    if (!target || !workspace || !options || !canSave || pending.current) return;
    const work = begin(); pending.current = true; setBusy(true); setNotification(quiet);
    try {
      if (target.kind === "book" && draft && book && feature) {
        const value = await service.apply({ doc_name: target.doc, expected_revision: book.revision,
          expected_workspace_revision: book.workspace.revision, selection: feature.selection(draft), source_language_override: draft.override }, work.signal);
        if (!work.current()) return;
        if (draftChanged(draft, value) || value.revision < book.revision)
          throw new PreferencesFailure("unknown");
        setBook(value); setWorkspace(value.workspace); setDraft(initialDraft(value));
        if (closeOnBookSave) leave();
        else notify("Reading mode saved and applied.", "success");
      } else if (target.kind === "workspace") {
        const value = await service.saveWorkspace(native, workspace.revision, work.signal);
        if (!work.current()) return;
        if (value.native_language !== native || value.revision < workspace.revision) throw new PreferencesFailure("unknown");
        setWorkspace(value); setNative(value.native_language);
        notify("Workspace preferences saved.", "success");
      }
    } catch (error) {
      if (!work.current()) return;
      const failure = error instanceof PreferencesFailure ? error : new PreferencesFailure("unknown");
      setRecovery(failure.kind === "unknown" || failure.kind === "conflict");
      notify(failure.issues.length ? `${failure.message} ${failure.issues.map(i => i.message).join(" ")}` : failure.message);
    } finally { if (work.current()) { pending.current = false; setBusy(false); } }
  }
  return { target, options, book, workspace, draft, native, status, busy, recovery, detail, notification, confirm, returnFocus,
    dirty, feature, problems, canSave, invalidNative, openBook, openWorkspace, close, choose, backToList, reload, save,
    changeDraft: (value: SettingsDraft) => { if (!pending.current && !recovery) setDraft(value); },
    changeNative: (value: string | null) => { if (!pending.current && !recovery) setNative(value); },
    closeNotification: () => setNotification(value => ({ ...value, open: false })),
    keepEditing: () => { destination.current = null; setConfirm(false); },
    discard: () => { const next = destination.current; destination.current = null; setConfirm(false); next?.(); },
  };
}
export type ReadingSettingsController = ReturnType<typeof useReadingSettingsController>;
