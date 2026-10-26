import { useEffect, useRef, useState } from "react";
import { smartCleanupService } from "../../services";
import type { CleanupConsent, CleanupOperation, CleanupPreflight } from "../../services/SmartCleanupService";
import { operationMessage, unavailableMessage } from "./model";
import { forgetCleanup, recoverCleanup, rememberCleanup } from "./recovery";

type Phase = "checking" | "confirm" | "running" | "blocked" | "unknown" | "resume" | "reload" | "done";
type State = { phase: Phase; message: string; preflight?: CleanupPreflight; expires?: number;
  result?: CleanupOperation; canCheck?: boolean };
const initial: State = { phase: "checking", message: "Checking cleanup readiness..." };
export interface SmartCleanupProps {
  open: boolean;
  docName: string;
  onClose: () => void;
  onReadingReady: (name: string, changed: boolean, signal: AbortSignal, reconcile?: boolean) => Promise<void>;
}

export function useSmartCleanup({ open, docName, onReadingReady }: SmartCleanupProps) {
  const [state, setState] = useState<State>(initial);
  const [now, setNow] = useState(Date.now());
  const generation = useRef(0), busy = useRef(false);
  const request = useRef<AbortController | null>(null);
  const consent = useRef<CleanupConsent | null>(null);
  const current = useRef({ open, docName, onReadingReady });
  current.current = { open, docName, onReadingReady };

  async function run(action: "open" | "fresh" | "lookup" | "execute" | "resume") {
    if (busy.current || !current.current.open || !docName.trim()) return;
    busy.current = true;
    const name = docName, id = ++generation.current;
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    const live = () => !controller.signal.aborted && id === generation.current
      && current.current.open && current.current.docName === name;
    const publish = (value: State) => { if (live()) setState(value); };
    const mutation = action === "execute" || action === "resume";
    try {
      // Re-read scoped handles on every action; no button may bypass recovery.
      const stored = recoverCleanup(name);
      consent.current = stored;
      if ((action === "open" || action === "fresh") && !stored) {
        publish(initial);
        const started = Date.now();
        const data = await smartCleanupService.preflight(name, controller.signal);
        if (!live()) return;
        setNow(Date.now());
        publish({ phase: data.execution_available ? "confirm" : "blocked", preflight: data,
          expires: started + (data.expires_in_seconds ?? 0) * 1000, canCheck: true,
          message: data.execution_available ? "Ready for smart cleanup." : unavailableMessage(data.reason) });
        return;
      }
      let result: CleanupOperation;
      if (action === "execute" && !stored) {
        const data = state.preflight;
        if (state.phase !== "confirm" || Date.now() >= (state.expires ?? 0) || !data?.operation_id || !data.confirmation_token) return;
        const handle = { operation_id: data.operation_id, confirmation_token: data.confirmation_token };
        rememberCleanup(name, handle);
        consent.current = handle;
        publish({ phase: "running", message: "Cleaning text and preparing reading..." });
        result = await smartCleanupService.execute(name, handle, controller.signal);
      } else if (stored) {
        publish({ phase: "checking", message: "Checking stored cleanup state..." });
        result = await smartCleanupService.lookup(name, stored.operation_id, controller.signal);
        if (!live()) return;
        // Resume requires an explicit click and a fresh authoritative layout state.
        if (action === "resume" && result.phase === "layout" && result.retry_action === "resume_layout") {
          publish({ phase: "running", message: "Finishing reading layout..." });
          result = await smartCleanupService.execute(name, stored, controller.signal);
        }
      } else {
        publish({ phase: "blocked", message: "No cleanup operation is available to recover.", canCheck: true });
        return;
      }
      if (!live()) return;
      const terminal = result.phase === "finished" || result.phase === "superseded";
      const reconcile = action !== "execute" || result.phase === "superseded"
        || ["source_changed", "reading_revision_changed"].includes(result.reason ?? "");
      if (result.reading_ready || (terminal && reconcile)) {
        try {
          await current.current.onReadingReady(name, result.outcome?.reading_content_changed === true,
            controller.signal, reconcile);
        } catch {
          publish({ phase: "reload", result, message: "The cleanup result was checked, but the reading view could not refresh. Retry reading without another cleanup." });
          return;
        }
      }
      if (!live()) return;
      // Forget only after certainty and the required view refresh both succeed.
      if (terminal) { forgetCleanup(name, result.operation_id); consent.current = null; }
      publish({ phase: terminal ? "done" : result.phase === "layout" ? "resume" : "unknown", result,
        message: operationMessage(result), canCheck: terminal });
    } catch (error) {
      if (live()) {
        const reason = error instanceof Error ? error.message : "";
        if (reason === "cleanup_confirmation_expired" && consent.current && action === "execute") {
          try { forgetCleanup(name, consent.current.operation_id); }
          catch { publish({ phase: "unknown", message: "The saved cleanup state could not be updated. Check stored state before continuing." }); return; }
          consent.current = null;
          publish({ phase: "blocked", canCheck: true, message: "Confirmation expired. Check again before starting cleanup." });
        } else if (consent.current) {
          publish({ phase: "unknown", message: "The cleanup result is not confirmed. Check stored state before continuing." });
        } else {
          publish({ phase: "blocked", canCheck: reason !== "cleanup_recovery_invalid",
            message: reason === "cleanup_recovery_invalid" ? "The saved cleanup handle cannot be verified. No new cleanup was started."
              : mutation ? "Browser recovery storage is unavailable. No cleanup was started." : unavailableMessage(reason) });
        }
      }
    } finally {
      if (id === generation.current) {
        busy.current = false;
      }
    }
  }

  useEffect(() => {
    busy.current = false; consent.current = null; setState(initial);
    if (open) void run("open");
    return () => { ++generation.current; request.current?.abort(); busy.current = false; };
  }, [open, docName]);
  useEffect(() => {
    if (!open || state.phase !== "confirm") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open, state.phase]);
  const expired = state.phase === "confirm" && now >= (state.expires ?? 0);
  return { ...state, message: expired ? "Confirmation expired. Check again before confirming." : state.message,
    canSubmit: open && state.phase === "confirm" && !expired,
    submit: () => run("execute"), check: () => run("lookup"), resume: () => run("resume"),
    fresh: () => run("fresh"), expired };
}
