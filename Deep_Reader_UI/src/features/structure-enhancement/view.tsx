import { useEffect, useRef, useState } from "react";
import { Alert, Button, CircularProgress, Dialog, DialogActions, DialogContent,
  DialogContentText, DialogTitle, Stack } from "@mui/material";
import { structureRepairService } from "../../services";
import type { EnhancementPreflightResponse, ReparseDocumentStructureResponse, TocHint } from "../../types/api";
import { capacityMessage, isCapacityRejection, sourceReadinessMessage } from "./capacity";
import { confirmedChange, type EnhancementChange } from "./outcome";

type Phase = "checking" | "confirm" | "pending" | "preparing" | "failed" | "unknown" | "reload" | "retry" | "done";
type Operation = { phase: Phase; data: EnhancementPreflightResponse | null; message: string;
  hint: TocHint; change?: EnhancementChange; layoutPrepared?: boolean; localCleared?: boolean };
const initial: Operation = { phase: "checking", data: null, message: "", hint: "auto" };

export function EnhancementDialog({ open, docName, tocHint, onClose, onCommitted, onPrepare, onPrepareLayout,
  onStructureChanged, onTitlesChanged }: {
  open: boolean; docName: string; tocHint: TocHint;
  onClose: () => void; onCommitted: (name: string) => Promise<void>; onPrepare: () => void;
  onPrepareLayout: (name: string) => Promise<void>;
  onStructureChanged: (name: string) => void; onTitlesChanged: (name: string) => Promise<void>;
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
    if (!change) return false;
    publish(name, { ...operation, change, phase: change === "unchanged" ? "done" : "reload",
      message: change === "unchanged" ? "No structure changes. Your interactions and drafts are retained."
        : change === "titles_only" ? "Titles updated. Your interactions and drafts are retained." : "Structure saved. Preparing reading layout..." });
    return true;
  }

  async function check(reconcile = false) {
    const name = docName;
    if (busy.current.has(name)) return;
    const id = ++generation.current;
    const previous = operations.current.get(name);
    const controller = new AbortController();
    request.current = controller;
    setState({ ...initial, data: previous?.data || null });
    let recover = false;
    try {
      if (reconcile && previous?.data?.confirmation_token) {
        const response = await structureRepairService.enhancementResult(name, previous.hint,
          previous.data.confirmation_token, controller.signal);
        if (generation.current !== id) return;
        if (response.status === "completed" && response.result && accept(name, previous, response.result)) recover = true;
        else if (response.status === "not_started" || (response.status === "failed" && response.retryable === true)) {
          publish(name, { ...previous, phase: "retry", message: "The operation can be retried. Existing data is retained." });
        } else if (response.status === "failed" && response.retryable === false) {
          publish(name, { ...previous, phase: "failed", message: "Enhancement was not applied. Check again before confirming a new attempt." });
        } else publish(name, { ...previous, phase: "unknown", message: response.status === "running"
          ? "Smart structuring is still running. Check again shortly."
          : "The earlier outcome cannot be confirmed. Existing local data is retained. A new attempt is blocked until the result is confirmed." });
      } else if (!reconcile) {
        const data = await structureRepairService.preflightEnhancement(name, tocHint, controller.signal);
        if (generation.current === id) publish(name, { phase: "confirm", data, hint: tocHint, message: "" });
      }
    } catch {
      if (generation.current === id) publish(name, { ...(previous || initial), phase: reconcile ? "unknown" : "failed",
        message: reconcile ? "Stored state could not be checked. The earlier outcome remains unknown."
          : "Could not check readiness. Check your connection and try again. No changes were made." });
    }
    if (recover && active(name)) await reload(name);
  }

  useEffect(() => {
    ++generation.current;
    const previous = operations.current.get(docName);
    if (open) {
      if (previous && previous.phase !== "confirm" && previous.phase !== "failed") setState(previous);
      else void check();
    }
    return () => { ++generation.current; request.current?.abort(); };
  }, [open, docName, tocHint]);

  useEffect(() => {
    if (!open) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open]);

  async function reload(name = docName) {
    let operation = operations.current.get(name);
    if (busy.current.has(name) || !operation?.change || operation.change === "unchanged" || !active(name)) return;
    busy.current.add(name);
    const progress = () => publish(name, operation!);
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
          await onPrepareLayout(name);
          operation = { ...operation, layoutPrepared: true, message: "Structure saved. Loading reading layout..." };
          progress();
        }
      }
      if (!active(name)) {
        publish(name, { ...operation, phase: "reload", message: "Structure updated. Resume reading when you return to this document." });
        return;
      }
      if (operation.change === "titles_only") await onTitlesChanged(name);
      else await onCommitted(name);
      operations.current.delete(name);
      if (active(name)) onClose();
    } catch {
      publish(name, { ...operation, phase: "reload", message: operation.change === "titles_only"
        ? "Titles saved, but label refresh failed. Retry to refresh them; your interactions and drafts are retained."
        : `Structure saved, but reading ${operation.layoutPrepared ? "reload" : "preparation"} failed. Retry to resume reading.` });
    } finally { busy.current.delete(name); }
  }

  const data = state.data;
  const currentTime = Math.max(now, Date.now());
  const valid = data?.ready === true && data.execution_available === true
    && data.capacity?.status === "fits" && data.capacity.full_source === true
    && data.cleanup_scope === "all_document_interactions_and_local_drafts"
    && data.cleanup_condition === "structure_changed_only"
    && Boolean(data.confirmation_token) && data.expires_at !== null && data.expires_at * 1000 > currentTime;

  async function submit() {
    const name = docName;
    if (busy.current.has(name) || !["confirm", "retry"].includes(state.phase) || !valid || !data?.confirmation_token
        || !data.expires_at || data.expires_at * 1000 <= Date.now()) return;
    busy.current.add(name);
    const operation = { ...state, data };
    publish(name, { ...operation, phase: "pending", message: "Enhancing structure..." });
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 120_000);
    let accepted = false;
    try {
      const result = await structureRepairService.enhance(name, operation.hint, data.confirmation_token, controller.signal);
      if (result.ok && accept(name, operation, result.payload)) accepted = true;
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
      publish(name, { ...operation, phase: "unknown", message: "Connection lost or timed out. The server may still be working. Existing local data is retained." });
    } finally { window.clearTimeout(timeout); busy.current.delete(name); }
    if (accepted && active(name)) await reload(name);
  }

  function close() {
    if (state.phase === "done") operations.current.delete(docName);
    onClose();
  }
  const waiting = ["pending", "checking", "preparing"].includes(state.phase);
  const cannotDismiss = ["pending", "reload", "preparing"].includes(state.phase);
  return <Dialog open={open} onClose={() => { if (!cannotDismiss) close(); }}
    TransitionProps={{ onEntered: () => cancelButton.current?.focus() }}
    fullWidth maxWidth="sm" aria-labelledby="enhancement-title">
    <DialogTitle id="enhancement-title">Smart structuring</DialogTitle>
    <DialogContent sx={{ overflowWrap: "anywhere" }}><Stack spacing={2}>
      <DialogContentText>{docName}</DialogContentText>
      <DialogContentText>If chapter or section divisions, levels, or grouping change, all existing insights, quizzes and answers, critical-thinking sessions, saved notes, and local drafts will be deleted, including book-level data. Title-only changes or no changes retain them. The source text is never changed.</DialogContentText>
      {data && <DialogContentText>Interactions affected only by a structural change: {data.interaction_count ?? "Unknown (not zero)"}. {data.includes_user_notes === true ? "Includes saved user notes." : data.includes_user_notes === null ? "Saved notes may also be present." : "No saved user notes reported."}</DialogContentText>}
      {state.phase === "confirm" && valid && <Alert severity="success" role="status">Ready for smart structuring.</Alert>}
      {waiting && <CircularProgress size={24} aria-label={state.phase === "pending" ? "Enhancing" : state.phase === "preparing" ? "Preparing reading layout" : "Checking readiness"} />}
      {state.message && <Alert severity={state.phase === "done" ? "success" : waiting || state.phase === "reload" ? "info" : "warning"}>{state.message}</Alert>}
      {state.phase === "confirm" && data && !data.ready && <Alert severity="warning">{sourceReadinessMessage(data.reason)}</Alert>}
      {state.phase === "confirm" && data?.ready && capacityMessage(data.capacity) && <Alert severity="warning">{capacityMessage(data.capacity)} No changes were made. You can keep reading.</Alert>}
      {state.phase === "confirm" && data?.ready && !capacityMessage(data.capacity) && !data.execution_available && <Alert severity="info">Smart structuring is currently unavailable. Try checking again later.</Alert>}
      {state.phase === "confirm" && data?.execution_available && data.cleanup_condition !== "structure_changed_only" && <Alert severity="warning">The service needs an update before smart structuring can start.</Alert>}
      {["confirm", "retry"].includes(state.phase) && data?.expires_at && data.expires_at * 1000 <= currentTime && <Alert severity="warning">Confirmation expired. Check again before confirming.</Alert>}
    </Stack></DialogContent>
    <DialogActions sx={{ flexWrap: "wrap", gap: 1 }}>
      <Button ref={cancelButton} autoFocus disabled={cannotDismiss} onClick={close}>{state.phase === "done" ? "Close" : "Cancel"}</Button>
      {state.phase === "confirm" && data && !data.ready && <Button onClick={() => { close(); onPrepare(); }}>Open Library</Button>}
      {!waiting && !["reload", "done"].includes(state.phase) && <Button onClick={() => void check(state.phase === "unknown")}>{state.phase === "unknown" ? "Check stored state" : "Check again"}</Button>}
      {state.phase === "reload" ? <Button onClick={() => void reload()}>Reload document</Button>
        : state.phase !== "done" && <Button variant="contained" disabled={!["confirm", "retry"].includes(state.phase) || !valid} onClick={() => void submit()}>{state.phase === "retry" ? "Retry smart structuring" : "Enhance"}</Button>}
    </DialogActions>
  </Dialog>;
}
