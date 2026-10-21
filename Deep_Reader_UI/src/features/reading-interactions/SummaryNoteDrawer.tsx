import { useEffect, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, Drawer, IconButton, Stack, TextField, Tooltip, Typography } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SaveIcon from "@mui/icons-material/Save";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import RefreshIcon from "@mui/icons-material/Refresh";
import type { ReadingInteractionTargetRequest } from "../../types/api";
import { SummaryNoteService, type NoteResponse, type SummaryResponse } from "../../services/SummaryNoteService";

const service = new SummaryNoteService();
export interface SummaryNoteSelection {
  kind: "summary" | "note";
  target: ReadingInteractionTargetRequest;
  breadcrumb: string[];
}

export function SummaryNoteDrawer({ selection, onClose, dirtyRef }: {
  selection: SummaryNoteSelection;
  onClose: () => void;
  dirtyRef: { current: boolean };
}) {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [note, setNote] = useState<NoteResponse | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef<AbortController | null>(null);
  const mutationRef = useRef<{ text: string; revision: number; id: string } | null>(null);
  const dirty = selection.kind === "note" && draft !== (note?.note?.text || "");
  const count = Array.from(draft).length;
  const limit = note?.max_chars ?? 10000;
  dirtyRef.current = dirty;

  useEffect(() => {
    const onUnload = (event: BeforeUnloadEvent) => {
      if (dirtyRef.current) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", onUnload);
    void execute("read");
    return () => {
      requestRef.current?.abort();
      dirtyRef.current = false;
      window.removeEventListener("beforeunload", onUnload);
    };
  }, []);

  async function execute(operation: "read" | "generate" | "refresh" | "save" | "delete") {
    if (busy) return;
    if (operation === "delete" && !window.confirm("Delete this note?")) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setBusy(true); setError("");
    try {
      if (selection.kind === "summary") {
        const result = await service.summary(selection.target, operation as "read" | "generate" | "refresh", controller.signal);
        if (controller.signal.aborted) return;
        if (result.envelope.status === "completed" || operation === "read") setSummary(result);
        if (!["completed", "not_generated"].includes(result.envelope.status)) {
          setError(result.envelope.reason === "summary_model_capacity_exceeded"
            ? "This text exceeds the model's capacity. Choose a smaller section."
            : result.envelope.status === "insufficient_content" ? "There is not enough readable content to summarize."
            : result.envelope.status === "stale_target" ? "The source has changed. Refresh the summary."
            : "Summary generation failed. Your saved summary is unchanged. Please retry.");
        }
      } else {
        const revision = note?.note?.revision ?? 0;
        if (operation === "save" && (!mutationRef.current || mutationRef.current.text !== draft || mutationRef.current.revision !== revision)) {
          mutationRef.current = { text: draft, revision, id: crypto.randomUUID() };
        }
        const fields = operation === "read" ? {} : {
          expected_revision: revision, target_token: note?.target_token,
          ...(operation === "save" ? { text: draft, mutation_id: mutationRef.current!.id } : {}),
        };
        const result = await service.note(selection.target, operation as "read" | "save" | "delete", fields, controller.signal);
        if (controller.signal.aborted) return;
        setNote(result); setDraft(result.note?.text || ""); mutationRef.current = null;
      }
    } catch (failure) {
      if (controller.signal.aborted) return;
      const message = failure instanceof Error ? failure.message : "";
      setError(message.includes("conflict") ? "This note was changed elsewhere. Your draft is kept; reopen to load the latest version."
        : message.includes("stale") ? "The document changed. Your draft is kept; reopen the current target."
        : "The request failed. Your draft and saved content are unchanged. Please retry.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return <Drawer anchor="right" open onClose={onClose}
    PaperProps={{ sx: { width: { xs: "100%", sm: 520 }, maxWidth: "100%" } }}>
    <Box role="region" aria-label={selection.kind === "summary" ? "Summary" : "Note"}
      sx={{ p: 3, overflowY: "auto", overflowWrap: "anywhere", height: "100%", boxSizing: "border-box" }}>
      <Stack spacing={2}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography component="h2" variant="h5">{selection.kind === "summary" ? "Summary" : "Note"}</Typography>
          <Tooltip title="Close"><IconButton aria-label="Close" onClick={onClose}><CloseIcon /></IconButton></Tooltip>
        </Stack>
        <Typography variant="body2" color="text.secondary">{selection.breadcrumb.join(" > ")}</Typography>
        {error && <Alert severity="error">{error}</Alert>}
        {busy && <CircularProgress size={24} aria-label="Loading" />}
        {selection.kind === "summary" ? <>
          <Button disabled={busy} startIcon={<RefreshIcon />} onClick={() => void execute(summary?.payload ? "refresh" : "generate")}>
            {summary?.payload ? "Refresh" : "Generate summary"}
          </Button>
          {summary?.payload ? <>
            <Typography component="h3" variant="h6">{summary.payload.headline}</Typography>
            <Typography sx={{ whiteSpace: "pre-wrap" }}>{summary.payload.body}</Typography>
            <Box component="ul" sx={{ pl: 3 }}>{summary.payload.key_points.map((point, index) => <li key={index}>{point}</li>)}</Box>
          </> : !busy && <Typography color="text.secondary">No saved summary.</Typography>}
        </> : <>
          {note?.source_stale && <Alert severity="warning">Reading content has changed. This saved note is retained.</Alert>}
          <TextField label="Note" multiline minRows={8} value={draft} disabled={busy || !note}
            onChange={event => setDraft(event.target.value)} error={count > limit}
            helperText={`${count.toLocaleString()} / ${limit.toLocaleString()}`} />
          <Stack direction="row" spacing={1}>
            <Button startIcon={<SaveIcon />} disabled={busy || !note || !dirty || !draft.trim() || count > limit}
              onClick={() => void execute("save")}>Save</Button>
            <Tooltip title="Delete note"><span><IconButton aria-label="Delete note" disabled={busy || !note?.note}
              onClick={() => void execute("delete")}><DeleteOutlineIcon /></IconButton></span></Tooltip>
            {!note && !busy && <Button onClick={() => void execute("read")}>Retry</Button>}
          </Stack>
        </>}
      </Stack>
    </Box>
  </Drawer>;
}
