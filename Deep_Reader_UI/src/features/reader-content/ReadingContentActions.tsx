import { useEffect, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
  IconButton, Stack, TextField, Tooltip, Typography } from "@mui/material";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import UndoIcon from "@mui/icons-material/Undo";
import FactCheckOutlinedIcon from "@mui/icons-material/FactCheckOutlined";
import CloseIcon from "@mui/icons-material/Close";
import { RestClient } from "../../services/RestClient";

export type ContentEditSelection = { blockId: string; action: "edit" | "delete" } | null;
type ReviewRegion = { id: string; page_index: number; reason: string; raw_candidate: string; status: string };
type ReadingBlock = { block_id: string; content: string; deleted: boolean };
type ReadingState = { revision: number; source_token: string; disposition: string; can_undo: boolean;
  max_chars: number; regions: ReviewRegion[]; blocks: ReadingBlock[] };
const client = new RestClient();
let dirtyGuard: (() => boolean) | null = null;
export function confirmReadingNavigation() { return dirtyGuard?.() ?? true; }

export function ReadingContentActions({ docName, selection, onCloseSelection, onChanged, onReviewState }: {
  docName: string; selection: ContentEditSelection; onCloseSelection: () => void; onChanged: () => void;
  onReviewState?: (required: boolean) => void;
}) {
  const [state, setState] = useState<ReadingState | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [text, setText] = useState("");
  const [saved, setSaved] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [loadedPreview, setLoadedPreview] = useState<string | null>(null);
  const [confirmRegion, setConfirmRegion] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const generation = useRef(0);
  const retry = useRef<{ command: string; id: string } | null>(null);
  const path = `/documents/${encodeURIComponent(docName)}/reading-content`;
  const dirty = selection?.action === "edit" && text !== saved;
  useEffect(() => { if (state) onReviewState?.(state.disposition === "needs_review"); }, [state, onReviewState]);
  useEffect(() => {
    const guard = () => !dirty || window.confirm("Discard unsaved changes?");
    dirtyGuard = guard;
    const unload = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", unload);
    return () => { if (dirtyGuard === guard) dirtyGuard = null; window.removeEventListener("beforeunload", unload); };
  }, [dirty]);
  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    setState(null); setError(""); setBusy(true); setPreview(null); setLoadedPreview(null); setConfirmRegion(null);
    client.requestJson<ReadingState>(`${path}${selection ? `?block_id=${encodeURIComponent(selection.blockId)}` : ""}`, { signal: controller.signal })
      .then(next => {
        if (current !== generation.current) return;
        setState(next);
        const block = next.blocks.find(b => b.block_id === selection?.blockId);
        setText(block?.content ?? ""); setSaved(block?.content ?? "");
        if (selection && (!block || block.deleted)) setError("This block changed. Close and reload the document.");
      }).catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Unable to load reading content."); })
      .finally(() => { if (current === generation.current) setBusy(false); });
    return () => { controller.abort(); ++generation.current; };
  }, [path, selection?.blockId, selection?.action, refresh]);

  function close() {
    if (busy || !confirmReadingNavigation()) return;
    setReviewOpen(false); setConfirmRegion(null); setError(""); onCloseSelection();
  }
  async function mutate(action: "edit" | "delete" | "undo" | "discard_region", regionId?: string) {
    if (!state || busy) return;
    const payload = { action, expected_revision: state.revision, source_token: state.source_token,
      ...(action === "edit" || action === "delete" ? { block_id: selection?.blockId } : {}),
      ...(action === "edit" ? { text } : {}), ...(regionId ? { region_id: regionId } : {}) };
    const command = JSON.stringify(payload);
    if (retry.current?.command !== command) retry.current = { command, id: crypto.randomUUID() };
    const current = generation.current;
    setBusy(true); setError("");
    try {
      const next = await client.requestJson<ReadingState>(path, { method: "POST", body: JSON.stringify({ ...payload, mutation_id: retry.current.id }) });
      if (current !== generation.current) return;
      setState(next); retry.current = null; setSaved(text); setConfirmRegion(null);
      onChanged();
      if (action === "edit" || action === "delete") onCloseSelection();
    } catch (e) { if (current === generation.current) setError(e instanceof Error ? e.message : "Change failed. Your draft is retained."); }
    finally { if (current === generation.current) setBusy(false); }
  }
  const count = Array.from(text).length;
  return <Box sx={{ px: 1, py: .5 }}>
    <Stack direction="row" alignItems="center" spacing={1}>
      {state?.disposition === "needs_review" && <Alert severity="warning" sx={{ flex: 1 }}>OCR needs review</Alert>}
      <Tooltip title="Review source"><span><IconButton aria-label="Review source" onClick={() => setReviewOpen(true)} disabled={busy}><FactCheckOutlinedIcon /></IconButton></span></Tooltip>
      <Tooltip title="Undo last content change"><span><IconButton aria-label="Undo last content change" disabled={busy || !state?.can_undo} onClick={() => void mutate("undo")}><UndoIcon /></IconButton></span></Tooltip>
      {error && !reviewOpen && !selection && <Alert severity="error">{error}</Alert>}
    </Stack>
    <Dialog open={reviewOpen || Boolean(selection)} onClose={close} fullWidth maxWidth="md" aria-labelledby="reading-content-dialog-title">
      <DialogTitle id="reading-content-dialog-title" sx={{ display: "flex", justifyContent: "space-between" }}>
        {selection ? (selection.action === "edit" ? "Edit block" : "Delete block?") : "Source review"}
        <IconButton aria-label="Close content editor" disabled={busy} onClick={close}><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ overflowWrap: "anywhere" }}>
        {busy && <CircularProgress size={22} aria-label="Loading content changes" />}
        {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
        {selection?.action === "edit" && <TextField label="Block text" multiline minRows={8} maxRows={18} fullWidth value={text} disabled={busy || !state}
          onChange={e => setText(e.target.value)} error={count > 10000} helperText={`${count} / 10000`} />}
        {selection?.action === "delete" && <><Alert severity="warning">Delete this block from the reading content? Related generated interactions will be invalidated. Notes are retained.</Alert><Typography sx={{ whiteSpace: "pre-wrap", mt: 2 }}>{text}</Typography></>}
        {!selection && state && <Stack spacing={2}>
          {state.regions.length === 0 && <Typography>No unresolved OCR regions recorded.</Typography>}
          {state.regions.map(region => <Box key={region.id} sx={{ borderBottom: "1px solid", borderColor: "divider", pb: 2 }}>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              <Typography>Page {region.page_index + 1} · {region.status === "unresolved" ? "Needs review" : "Ignored"}</Typography>
              <Button onClick={() => { if (preview !== region.id) setLoadedPreview(null); setPreview(region.id); setError(""); }}>Original</Button>
              {region.status === "unresolved" && <Button color="warning" disabled={busy || loadedPreview !== region.id} onClick={() => setConfirmRegion(region.id)}>Ignore region</Button>}
            </Stack>
            {region.raw_candidate && <Typography sx={{ whiteSpace: "pre-wrap", maxHeight: 180, overflow: "auto" }}>{region.raw_candidate}</Typography>}
            {preview === region.id && <Box component="img" alt={`Original page ${region.page_index + 1} region`} sx={{ maxWidth: "100%", maxHeight: 550, objectFit: "contain" }}
              src={`/api/documents/${encodeURIComponent(docName)}/source-review/${encodeURIComponent(region.id)}/image`}
              onLoad={() => setLoadedPreview(region.id)}
              onError={() => { setPreview(null); setLoadedPreview(null); setError("Original source is unavailable or changed. Reload before continuing."); }} />}
          </Box>)}
        </Stack>}
      </DialogContent>
      <DialogActions>
        <Button onClick={close} disabled={busy}>Cancel</Button>
        {error && <Button disabled={busy} onClick={() => { if (confirmReadingNavigation()) setRefresh(x => x + 1); }}>Reload</Button>}
        {selection?.action === "edit" && <Button startIcon={<EditOutlinedIcon />} disabled={busy || !state || !text.trim() || count > 10000} onClick={() => void mutate("edit")}>Save</Button>}
        {selection?.action === "delete" && <Button color="error" startIcon={<DeleteOutlineIcon />} disabled={busy || !state} onClick={() => void mutate("delete")}>Delete block</Button>}
      </DialogActions>
    </Dialog>
    <Dialog open={Boolean(confirmRegion)} onClose={() => !busy && setConfirmRegion(null)}>
      <DialogTitle>Ignore this OCR region?</DialogTitle>
      <DialogContent>The unreadable region will be excluded from reading and generation. The original image is retained. This can be undone.</DialogContent>
      <DialogActions><Button disabled={busy} onClick={() => setConfirmRegion(null)}>Cancel</Button><Button disabled={busy} color="warning" onClick={() => void mutate("discard_region", confirmRegion!)}>Ignore region</Button></DialogActions>
    </Dialog>
  </Box>;
}
