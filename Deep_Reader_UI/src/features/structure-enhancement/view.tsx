import { useEffect, useRef, useState } from "react";
import { Alert, Button, CircularProgress, Dialog, DialogActions, DialogContent,
  DialogContentText, DialogTitle, Stack } from "@mui/material";
import { structureRepairService } from "../../services";
import type { EnhancementPreflightResponse, ReparseDocumentStructureResponse, TocHint } from "../../types/api";
import { capacityMessage, isCapacityRejection, sourceReadinessMessage } from "./capacity";
import { cleanupConsentValid, cleanupMessage, confirmedChange, type EnhancementChange } from "./outcome";
import CleaningServicesOutlinedIcon from "@mui/icons-material/CleaningServicesOutlined";

type Phase = "checking" | "confirm" | "pending" | "preparing" | "failed" | "unknown" | "reload" | "resume" | "retry" | "done";
type Operation = { phase: Phase; data: EnhancementPreflightResponse | null; message: string;
  hint: TocHint; change?: EnhancementChange; layoutPrepared?: boolean; localCleared?: boolean;
  result?: Partial<ReparseDocumentStructureResponse>; contentReloaded?: boolean; submitted?: boolean };
const initial: Operation = { phase: "checking", data: null, message: "", hint: "auto" };

export function EnhancementDialog({ open, docName, tocHint, onClose, onCommitted, onPrepare, onPrepareLayout,
  onStructureChanged, onTitlesChanged, onReadingReady, onSmartCleanup }: {
  open: boolean; docName: string; tocHint: TocHint;
  onClose: () => void; onCommitted: (name: string, signal?: AbortSignal) => Promise<void>; onPrepare: () => void;
  onPrepareLayout: (name: string, signal?: AbortSignal) => Promise<void>;
  onStructureChanged: (name: string) => void; onTitlesChanged: (name: string, signal?: AbortSignal) => Promise<void>;
  onReadingReady?: (name: string, contentChanged: boolean, signal?: AbortSignal) => Promise<void>;
  onSmartCleanup?: () => void;
}) {
  const [state, setState] = useState<Operation>(initial);
  const [now, setNow] = useState(Date.now());
  const generation = useRef(0);
  const busy = useRef(new Set<string>());
  const cancelButton = useRef<HTMLButtonElement>(null);
  // Bind reconciliation and retries to the original consent, including its TOC hint.
  const operations = useRef(new Map<string, Operation>());
  const request = useRef<AbortController | null>(null);
  const current = useRef({ open, docName });
  current.current = { open, docName };
  const active = (name: string) => current.current.open && current.current.docName === name;

  function publish(name: string, value: Operation) {
    operations.current.set(name, value);
    if (active(name)) setState(value);
  }

  function accept(name: string, operation: Operation, result: Partial<ReparseDocumentStructureResponse>) {
    const change = confirmedChange(result, name);
    if (!change || (operation.data?.ocr_cleanup?.enabled && !result.phase)) return false;
    if (result.phase) {
      const superseded = result.phase === "superseded";
      const needsReload = !superseded && (change !== "unchanged" || result.cleanup?.reading_content_changed === true || Boolean(onReadingReady));
      const phase = superseded ? "done" : result.phase === "layout" ? "resume" : needsReload ? "reload" : "done";
      const message = superseded ? "This result belongs to an earlier reading version. Current content and edits are retained."
        : result.phase === "layout" ? "Changes are saved. Reading layout still needs to finish. Resume without repeating smart structuring."
        : `${change === "unchanged" ? "No structure changes." : change === "titles_only" ? "Titles updated." : "Structure saved."} ${cleanupMessage(result.cleanup)}${result.review_required ? " Some extracted text still needs review." : ""}`;
      publish(name, { ...operation, change, result, phase, message, layoutPrepared: result.phase === "complete" });
      return true;
    }
    publish(name, { ...operation, change, phase: change === "unchanged" ? "done" : "reload",
      message: change === "unchanged" ? "No structure changes. Your interactions and drafts are retained."
        : change === "titles_only" ? "Titles updated. Your interactions and drafts are retained." : "Structure saved. Preparing reading layout..." });
    return true;
  }

  async function check(reconcile = false, preflight = true) {
    const name = docName;
    if (!name.trim() || busy.current.has(name)) return;
    const id = ++generation.current;
    const previous = operations.current.get(name);
    const controller = new AbortController();
    request.current?.abort();
    request.current = controller;
    setState({ ...initial, data: previous?.data || null });
    let recover = false;
    try {
      const activity = await structureRepairService.enhancementActivity(name, controller.signal);
      if (generation.current !== id) return;
      if (activity === "running") {
        publish(name, { ...(previous || initial), phase: "unknown",
          message: "Document processing is still running. Check again shortly." });
        return;
      }
      if (!preflight) return;
      if (reconcile && previous?.submitted && previous.data?.confirmation_token) {
        const response = await structureRepairService.enhancementResult(name, previous.hint,
          previous.data.confirmation_token, controller.signal, previous.data.ocr_cleanup?.enabled === true);
        if (generation.current !== id) return;
        if (response.status === "completed" && response.result && accept(name, previous, response.result)) recover = true;
        else if (response.status === "not_started" || (response.status === "failed" && response.retryable === true)) {
          publish(name, { ...previous, phase: "retry", message: "The operation can be retried. Existing data is retained." });
        } else if (response.status === "failed" && response.retryable === false) {
          publish(name, { ...previous, phase: "failed", message: "Enhancement was not applied. Check again before confirming a new attempt." });
        } else if (response.status === "running") {
          publish(name, { ...previous, phase: "unknown", message: "Smart structuring is still running. Check again shortly." });
        } else {
          // No live work remains. New consent is for current source, not a replay of the old token.
          const data = await structureRepairService.preflightEnhancement(name, tocHint, controller.signal);
          if (generation.current === id) publish(name, { phase: "confirm", data, hint: tocHint, message: "" });
        }
      } else {
        const data = await structureRepairService.preflightEnhancement(name, tocHint, controller.signal);
        if (generation.current === id) publish(name, { phase: "confirm", data, hint: tocHint, message: "" });
      }
    } catch {
      const submitted = reconcile && previous?.submitted;
      if (generation.current === id) publish(name, { ...(previous || initial), phase: submitted ? "unknown" : "failed",
        message: submitted ? "Stored state could not be checked. The earlier outcome remains unknown."
          : "Could not check readiness. Check your connection and try again. No changes were made." });
    }
    if (recover && active(name)) await reload(name);
  }

  useEffect(() => {
    ++generation.current;
    const previous = operations.current.get(docName);
    if (open) {
      if (previous && previous.phase !== "confirm" && previous.phase !== "failed") setState(previous);
      else {
        void check(Boolean(previous?.data));
      }
      if (previous?.phase === "unknown") void check(Boolean(previous.data));
    } else {
      // React mount also covers a page reload; never restore browser-persisted operations.
      void check(false, false);
    }
    return () => {
      ++generation.current;
      request.current?.abort();
      const pending = operations.current.get(docName);
      if (pending?.phase === "pending") {
        operations.current.set(docName, { ...pending, phase: "unknown",
          message: "The server may still be working. Check stored state before continuing." });
        busy.current.delete(docName);
      } else if (pending?.phase === "preparing") {
        operations.current.set(docName, { ...pending, phase: "reload",
          message: "Changes are saved. Resume reading refresh when you return." });
        busy.current.delete(docName);
      }
    };
  }, [open, docName, tocHint]);

  useEffect(() => {
    if (!open) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open]);

  async function reload(name = docName) {
    let operation = operations.current.get(name);
    if (busy.current.has(name) || !operation?.change || operation.phase !== "reload" || !active(name)) return;
    busy.current.add(name);
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const id = ++generation.current;
    const live = () => generation.current === id && !controller.signal.aborted && active(name);
    const progress = () => { if (live()) publish(name, operation!); };
    try {
      operation = { ...operation, phase: "preparing" };
      progress();
      if (operation.change === "structure_changed") {
        if (!operation.localCleared) {
          onStructureChanged(name);
          operation = { ...operation, localCleared: true };
          progress();
        }
        if (!operation.layoutPrepared) {
          await onPrepareLayout(name, controller.signal);
          if (!live()) return;
          operation = { ...operation, layoutPrepared: true, message: "Structure saved. Loading reading layout..." };
          progress();
        }
      }
      if (!live()) return;
      if (operation.result?.phase && operation.change !== "structure_changed" && !operation.contentReloaded) {
        if (!active(name)) throw new Error("Document changed.");
        if (!onReadingReady && operation.result.cleanup?.reading_content_changed) throw new Error("Reading refresh unavailable.");
        await onReadingReady?.(name, operation.result.cleanup?.reading_content_changed === true, controller.signal);
        if (!live()) return;
        operation = { ...operation, contentReloaded: true };
      }
      if (operation.change === "titles_only") await onTitlesChanged(name, controller.signal);
      else if (operation.change === "structure_changed") await onCommitted(name, controller.signal);
      if (!live()) return;
      if (operation.result?.phase) {
        publish(name, { ...operation, phase: "done", message:
          `${operation.change === "unchanged" ? "No structure changes." : operation.change === "titles_only" ? "Titles updated." : "Structure saved."} ${cleanupMessage(operation.result.cleanup)}${operation.result.review_required ? " Some extracted text still needs review." : ""}` });
        return;
      }
      operations.current.delete(name);
      if (active(name)) onClose();
    } catch {
      if (live()) publish(name, { ...operation, phase: "reload", message: operation.result?.phase
        ? "Changes saved, but reading refresh failed. Retry the refresh; smart structuring will not run again."
        : operation.change === "titles_only"
        ? "Titles saved, but label refresh failed. Retry to refresh them; your interactions and drafts are retained."
        : `Structure saved, but reading ${operation.layoutPrepared ? "reload" : "preparation"} failed. Retry to resume reading.` });
    } finally { if (generation.current === id) busy.current.delete(name); }
  }

  const data = state.data;
  const currentTime = Math.max(now, Date.now());
  const valid = data?.ready === true && data.execution_available === true
    && data.capacity?.status === "fits" && data.capacity.full_source === true
    && data.cleanup_scope === "all_document_interactions_and_local_drafts"
    && data.cleanup_condition === "structure_changed_only"
    && cleanupConsentValid(data)
    && Boolean(data.confirmation_token) && data.expires_at !== null && data.expires_at * 1000 > currentTime;

  async function submit() {
    const name = docName;
    const resume = state.phase === "resume";
    if (busy.current.has(name) || !["confirm", "retry", "resume"].includes(state.phase) || !data?.confirmation_token
        || (!resume && (!valid || !data.expires_at || data.expires_at * 1000 <= Date.now()))) return;
    busy.current.add(name);
    const operation = { ...state, data, submitted: true };
    publish(name, { ...operation, phase: "pending", message: resume ? "Finishing reading layout..."
      : data.ocr_cleanup?.enabled ? "Structuring, checking OCR cleanup, and preparing reading..." : "Enhancing structure..." });
    const controller = new AbortController();
    request.current?.abort();
    request.current = controller;
    const id = ++generation.current;
    const live = () => generation.current === id && !controller.signal.aborted && active(name);
    let accepted = false;
    try {
      const result = await structureRepairService.enhance(name, operation.hint, data.confirmation_token, controller.signal,
        data.ocr_cleanup?.enabled === true);
      if (!live()) return;
      if (result.ok && accept(name, operation, result.payload)) accepted = true;
      else if (resume) {
        publish(name, { ...operation, phase: "unknown", message: "Changes were already saved, but reading recovery could not be confirmed. Check stored state before continuing." });
      }
      else if (isCapacityRejection(result.status, result.payload.detail, result.payload.capacity)) {
        publish(name, { ...operation, phase: "failed", data: { ...data, capacity: result.payload.capacity,
          execution_available: false, confirmation_token: null },
          message: `${capacityMessage(result.payload.capacity)} No changes were made. Check again before confirming.` });
      } else if (["enhancement_in_progress", "enhancement_outcome_unknown", "enhancement_execution_failed"].includes(result.payload.detail || "")) {
        publish(name, { ...operation, phase: "unknown", message: "Check stored state before retrying this operation." });
      } else if (result.payload.success === false && result.payload.doc_name === name && result.payload.retryable === true) {
        publish(name, { ...operation, phase: "retry", message: "The service was temporarily unavailable. Retry smart structuring; existing data is retained." });
      } else if (result.status === 409 || result.status === 422
          || (result.payload.success === false && result.payload.doc_name === name && result.payload.retryable === false)) {
        publish(name, { ...operation, phase: "failed", message: "Enhancement was not applied. Existing data is retained. Check again before confirming." });
      } else publish(name, { ...operation, phase: "unknown", message: "The server did not confirm the outcome. Check stored state before any retry." });
    } catch {
      if (live()) publish(name, { ...operation, phase: "unknown", message: "Connection lost or timed out. The server may still be working. Existing local data is retained." });
    } finally { if (generation.current === id) busy.current.delete(name); }
    if (accepted && live()) await reload(name);
  }

  function close() {
    if (["done", "failed", "confirm"].includes(state.phase)) operations.current.delete(docName);
    onClose();
  }
  const waiting = ["pending", "checking", "preparing"].includes(state.phase);
  return <Dialog open={open} onClose={close}
    TransitionProps={{ onEntered: () => cancelButton.current?.focus() }}
    fullWidth maxWidth="sm" aria-labelledby="enhancement-title">
    <DialogTitle id="enhancement-title">Smart structuring</DialogTitle>
    <DialogContent sx={{ overflowWrap: "anywhere" }}><Stack spacing={2}>
      <DialogContentText>{docName}</DialogContentText>
      <DialogContentText>If chapter or section divisions, levels, or grouping change, all existing insights, quizzes and answers, critical-thinking sessions, saved notes, and local drafts will be deleted, including book-level data. Title-only changes or no changes retain them. The source text is never changed.</DialogContentText>
      {data?.ocr_cleanup?.enabled && <DialogContentText>For verified OCR text, this confirmation also permits removing clearly unreadable fragments from the reading copy. Cleanup alone invalidates generated interactions but retains saved notes and their drafts. Original text is retained and cleanup can be undone. Uncertain regions may still require review.</DialogContentText>}
      {data && <DialogContentText>Interactions affected only by a structural change: {data.interaction_count ?? "Unknown (not zero)"}. {data.includes_user_notes === true ? "Includes saved user notes." : data.includes_user_notes === null ? "Saved notes may also be present." : "No saved user notes reported."}</DialogContentText>}
      {state.phase === "confirm" && valid && <Alert severity="success" role="status">Ready for smart structuring.</Alert>}
      {waiting && <CircularProgress size={24} aria-label={state.phase === "pending" ? "Enhancing" : state.phase === "preparing" ? "Preparing reading layout" : "Checking readiness"} />}
      {state.message && <Alert severity={state.phase === "done" ? (["failed", "partial", "skipped"].includes(state.result?.cleanup?.status || "") || state.result?.review_required || state.result?.phase === "superseded" ? "warning" : "success") : waiting || state.phase === "reload" ? "info" : "warning"}>{state.message}</Alert>}
      {state.phase === "confirm" && data && !data.ready && <Alert severity="warning">{sourceReadinessMessage(data.reason)}</Alert>}
      {state.phase === "confirm" && data?.ready && capacityMessage(data.capacity) && <Alert severity="warning">{capacityMessage(data.capacity)} No changes were made. You can keep reading.</Alert>}
      {state.phase === "confirm" && data?.ready && !capacityMessage(data.capacity) && !data.execution_available && <Alert severity="info">Smart structuring is currently unavailable. Try checking again later.</Alert>}
      {state.phase === "confirm" && data?.execution_available && data.cleanup_condition !== "structure_changed_only" && <Alert severity="warning">The service needs an update before smart structuring can start.</Alert>}
      {state.phase === "confirm" && data?.ocr_cleanup && !cleanupConsentValid(data) && <Alert severity="warning">Cleanup permissions could not be verified. Update the service before continuing.</Alert>}
      {["confirm", "retry"].includes(state.phase) && data?.expires_at && data.expires_at * 1000 <= currentTime && <Alert severity="warning">Confirmation expired. Check again before confirming.</Alert>}
    </Stack></DialogContent>
    <DialogActions sx={{ flexWrap: "wrap", gap: 1 }}>
      {state.phase === "done" && state.result?.phase === "complete" && onSmartCleanup
        && ["failed", "partial", "skipped"].includes(state.result.cleanup?.status ?? "") &&
        <Button className="enhancement-smart-cleanup" startIcon={<CleaningServicesOutlinedIcon />}
          onClick={onSmartCleanup}>Smart cleanup</Button>}
      <Button ref={cancelButton} autoFocus onClick={close}>{state.phase === "done" ? "Close" : "Cancel"}</Button>
      {state.phase === "confirm" && data && !data.ready && <Button onClick={() => { close(); onPrepare(); }}>Open Library</Button>}
      {!waiting && !["reload", "done", "resume"].includes(state.phase) && <Button onClick={() => void check(state.phase === "unknown")}>{state.phase === "unknown" ? "Check stored state" : "Check again"}</Button>}
      {state.phase === "reload" ? <Button onClick={() => void reload()}>Reload document</Button>
        : state.phase !== "done" && <Button variant="contained" disabled={state.phase !== "resume" && (!["confirm", "retry"].includes(state.phase) || !valid)} onClick={() => void submit()}>{state.phase === "resume" ? "Resume reading" : state.phase === "retry" ? "Retry smart structuring" : "Enhance"}</Button>}
    </DialogActions>
  </Dialog>;
}
