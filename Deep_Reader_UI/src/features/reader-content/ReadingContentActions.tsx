import { useEffect, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
  IconButton, Paper, Stack, Tab, Tabs, TextField, Tooltip, Typography } from "@mui/material";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import UndoIcon from "@mui/icons-material/Undo";
import FactCheckOutlinedIcon from "@mui/icons-material/FactCheckOutlined";
import CloseIcon from "@mui/icons-material/Close";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RefreshIcon from "@mui/icons-material/Refresh";
import { readingContentService, ReadingMutationRejected, type ReadingState, type ReadingMutation, type ReadingRefresh, type ReviewBlockCommand, type ReadingBlock } from "../../services/ReadingContentService";
import { SourceReviewBlockEditor, type ReviewBlockMode } from "./SourceReviewBlockRepair";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { CleanupAudit } from "./CleanupAudit";
import { sameReadingSnapshot } from "./sourceReviewModel";
import { SourceReviewBlocks, SourceReviewManualChange } from "./SourceReviewBlocks";
import type { ReviewNavigate } from "./useSourceReviewBlocks";
import type { CleanupCommand } from "./useCleanupHistory";
import { mergeCleanupChange } from "./cleanupHistoryModel";

export type ContentEditSelection = { blockId: string; action: "edit" | "delete"; regionId?: string } | null;
let dirtyGuard: (() => boolean) | null = null;
export function confirmReadingNavigation() { return dirtyGuard?.() ?? true; }

export function ReadingContentActions({ docName, selection, onCloseSelection, onChanged, onRefreshBlocks, onReviewState, onNavigate, refreshKey = 0 }: {
  docName: string; selection: ContentEditSelection; onCloseSelection: () => void;
  onChanged: (signal?: AbortSignal, clearedBlockId?: string) => void | Promise<void>;
  onRefreshBlocks: (refresh: ReadingRefresh, signal: AbortSignal) => Promise<void>;
  onReviewState?: (required: boolean) => void;
  onNavigate: ReviewNavigate;
  refreshKey?: number;
}) {
  const [state, setState] = useState<ReadingState | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [reviewTab, setReviewTab] = useState<"history" | "regions">("history");
  const reviewScroll = useRef<HTMLDivElement | null>(null);
  const tabScroll = useRef({ history: 0, regions: 0 });
  useEffect(() => { setReviewTab("history"); tabScroll.current = { history: 0, regions: 0 }; }, [docName]);
  useEffect(() => { if (reviewScroll.current) reviewScroll.current.scrollTop = tabScroll.current[reviewTab]; }, [reviewTab]);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [regionActivation, setRegionActivation] = useState(0);
  const [navigated, setNavigated] = useState(false);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [confirmEdit, setConfirmEdit] = useState(false);
  const [text, setText] = useState("");
  const [saved, setSaved] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [loadedPreview, setLoadedPreview] = useState<string | null>(null);
  const [confirmRegion, setConfirmRegion] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [auditOffset, setAuditOffset] = useState(0);
  const [cleanupDirty, setCleanupDirty] = useState(false);
  const [regionDirty, setRegionDirty] = useState(false);
  const [regionEditor, setRegionEditor] = useState<{ id: string; block: ReadingBlock; mode: ReviewBlockMode } | null>(null);
  const regionReturnFocus = useRef<{ element: HTMLElement; scrollTop: number } | null>(null);
  useEffect(() => {
    if (regionEditor || !reviewOpen) return;
    const saved = regionReturnFocus.current;
    regionReturnFocus.current = null;
    if (!saved) return;
    const frame = requestAnimationFrame(() => {
      const target = saved.element.isConnected ? saved.element : document.querySelector<HTMLElement>(".source-review-blocks-toggle, #source-review-regions-tab");
      target?.focus({ preventScroll: true });
      if (reviewScroll.current) reviewScroll.current.scrollTop = saved.scrollTop;
    });
    return () => cancelAnimationFrame(frame);
  }, [regionEditor, reviewOpen]);
  const [regionPage, setRegionPage] = useState(0);
  useEffect(() => { setRegionEditor(null); setRegionDirty(false); setRegionPage(0); retry.current = null; }, [docName]);
  useEffect(() => { if (state) setRegionPage(value => Math.min(value, Math.max(0, Math.ceil(state.regions.length / 20) - 1))); }, [state?.regions.length]);
  const selectionKey = JSON.stringify([docName, selection?.blockId, selection?.action, reviewOpen, refresh, refreshKey, auditOffset]);
  const [readingRefreshFailed, setReadingRefreshFailed] = useState(false);
  const mutating = useRef(false);
  const generation = useRef(0);
  const retry = useRef<{ command: string; id: string } | null>(null);
  const mutationRequest = useRef<AbortController | null>(null);
  const pendingRefresh = useRef<{ kind: "blocks" | "layout" | "history"; binding?: ReadingRefresh | null;
    clearedBlockId?: string; snapshot?: ReadingState }>({ kind: "history" });
  const dirty = cleanupDirty || regionDirty || (selection?.action === "edit" && text !== saved);
  const ready = Boolean(state) && loadedKey === selectionKey && !busy
    && (!selection || Boolean(state?.blocks.some(block => block.block_id === selection.blockId && !block.deleted)));
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
    setState(null); setLoadedKey(null); setError(""); setBusy(true); setPreview(null); setLoadedPreview(null); setConfirmRegion(null); setConfirmEdit(false);
    const query = new URLSearchParams();
    if (selection) query.set("block_id", selection.blockId);
    else if (reviewOpen) { query.set("include_cleanup_audit", "true"); query.set("cleanup_offset", String(auditOffset)); }
    // Let synchronous effect cleanup (including StrictMode replay) cancel before dispatch.
    Promise.resolve().then(() => {
      if (controller.signal.aborted || current !== generation.current) return null;
      return readingContentService.read(docName, query, controller.signal);
    })
      .then(next => {
        if (!next || controller.signal.aborted || current !== generation.current) return;
        setState(next); setLoadedKey(selectionKey); retry.current = null;
        const block = next.blocks.find(b => b.block_id === selection?.blockId);
        setText(block?.content ?? ""); setSaved(block?.content ?? "");
        if (selection && (!block || block.deleted)) setError("This block changed. Close and reload the document.");
      }).catch(e => { if (!controller.signal.aborted && current === generation.current) setError(e instanceof Error ? e.message : "Unable to load reading content."); })
      .finally(() => { if (current === generation.current) setBusy(false); });
    return () => { controller.abort(); mutationRequest.current?.abort(); ++generation.current; };
  }, [docName, selection?.blockId, selection?.action, refresh, refreshKey, reviewOpen, auditOffset]);

  useEffect(() => {
    if (!reviewOpen || reviewTab !== "history" || !state || state.cleanup_audit || loadedKey !== selectionKey) return;
    const controller = new AbortController();
    const expected = state;
    readingContentService.read(docName, new URLSearchParams({ include_cleanup_audit: "true",
      cleanup_offset: String(auditOffset) }), controller.signal).then(next => {
      if (controller.signal.aborted) return;
      if (!sameReadingSnapshot(expected, next)) throw new Error("Content changed. Refresh source review.");
      setState(current => current && sameReadingSnapshot(current, next)
        ? { ...current, cleanup_audit: next.cleanup_audit } : current);
    }).catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Unable to load cleanup history."); });
    return () => controller.abort();
  }, [docName, reviewOpen, reviewTab, state?.revision, state?.cleanup_audit, loadedKey, selectionKey, auditOffset]);

  function close() {
    if ((busy && Boolean(selection)) || mutating.current || !confirmReadingNavigation()) return;
    setReviewOpen(false); setConfirmRegion(null); setPreview(null); setLoadedPreview(null); setError(""); onCloseSelection();
    setRegionEditor(null); setRegionDirty(false);
  }
  async function finishRefresh(signal: AbortSignal, current: number) {
    const pending = pendingRefresh.current;
    if (pending.kind === "blocks") {
      if (!pending.binding) throw new Error("Targeted refresh is unavailable. Reload the document.");
      await onRefreshBlocks(pending.binding, signal);
    } else if (pending.kind === "layout") await onChanged(signal, pending.clearedBlockId);
    if (signal.aborted || current !== generation.current) throw new DOMException("Aborted", "AbortError");
    if (pending.snapshot && reviewOpen) {
      const next = await readingContentService.read(docName, new URLSearchParams({
        include_cleanup_audit: "true", cleanup_offset: String(auditOffset) }), signal);
      if (signal.aborted || current !== generation.current) throw new DOMException("Aborted", "AbortError");
      if (!sameReadingSnapshot(next, pending.snapshot)) throw new Error("Content changed. Reload the document.");
      setState(next);
    }
  }
  async function mutate(action: ReadingMutation["action"], regionId?: string, discardInteractions?: boolean, cleanup?: CleanupCommand,
    regionCommand?: ReviewBlockCommand) {
    if (!state || !ready || mutating.current || readingRefreshFailed) return;
    if (action === "undo" && !confirmReadingNavigation()) return;
    mutating.current = true;
    let payload = { action, expected_revision: state.revision, source_token: state.source_token,
      ...(action === "edit" || action === "delete" ? { block_id: selection?.blockId } : {}),
      ...(action === "edit" ? { text, discard_interactions: discardInteractions } : {}),
      ...(cleanup ?? {}),
      ...(regionCommand ?? {}),
      ...((regionId ?? selection?.regionId) ? { region_id: regionId ?? selection?.regionId } : {}) };
    if (retry.current) {
      const previous = JSON.parse(retry.current.command) as typeof payload;
      if (previous.action.startsWith("review_block_")) {
        if (["action", "region_id", "block_id", "text", "discard_interactions"].some(key =>
          previous[key as keyof typeof payload] !== payload[key as keyof typeof payload])) {
          mutating.current = false;
          setError("The earlier region change is unconfirmed. Retry that same action before making another change.");
          return false;
        }
        payload = previous;
      }
    }
    const command = JSON.stringify(payload);
    if (retry.current?.command !== command) retry.current = { command, id: crypto.randomUUID() };
    const current = generation.current;
    const controller = new AbortController(); mutationRequest.current = controller;
    setBusy(true); setError("");
    try {
      const next = await readingContentService.mutate(docName, { ...payload, mutation_id: retry.current.id }, controller.signal);
      if (controller.signal.aborted || current !== generation.current) return;
      const mutationId = retry.current.id;
      const regionAction = action.startsWith("review_block_");
      const regionUndo = action === "undo" && ["region_apply", "region_delete", "review_block_edit", "review_block_delete"].includes(state.undo_action ?? "");
      const historyOnly = action === "cleanup_edit" || action === "cleanup_delete" || action === "review_block_keep";
      const bound = next.refresh?.mutation_id === mutationId && next.refresh.revision === next.revision
        && next.refresh.source_token === next.source_token ? next.refresh : null;
      if (regionAction && (!bound?.scope_current || next.source_token !== state.source_token
          || next.document_id !== state.document_id || (next.structure_version ?? null) !== (state.structure_version ?? null))) {
        retry.current = null;
        throw new Error("Content changed. Reload before continuing.");
      }
      // Merge committed history rows in place; no review GET or body refresh is needed.
      if (historyOnly && bound?.cleanup_change && cleanup) {
        if (next.source_token !== state.source_token || next.document_id !== state.document_id
          || bound.document_id !== next.document_id
          || (next.structure_version ?? null) !== (state.structure_version ?? null) || next.revision !== state.revision + 1) {
          throw new Error("Content changed. Reload before continuing.");
        }
        const history = mergeCleanupChange(state.cleanup_audit, bound, cleanup.cleanup_record_id, action === "cleanup_delete");
        setState({ ...state, revision: next.revision, can_undo: next.can_undo, undo_action: next.undo_action,
          cleanup_audit: history });
        retry.current = null; setCleanupDirty(false);
        return true;
      }
      setState({ ...next, cleanup_audit: (regionAction && !historyOnly) || regionUndo ? undefined : state.cleanup_audit });
      retry.current = null; setSaved(text); setConfirmRegion(null); setConfirmEdit(false);
      if (!next.regions.some(region => region.id === preview)) {
        setPreview(null); setLoadedPreview(null);
      }
      if (cleanup) setCleanupDirty(false);
      if (bound?.scope_current && bound.body_changed === false && next.revision === state.revision) {
        pendingRefresh.current = { kind: "history", binding: bound };
        setReadingRefreshFailed(false);
        if (action === "edit") onCloseSelection();
        return true;
      }
      pendingRefresh.current = { kind: historyOnly ? "history"
        : bound || regionAction || regionUndo || action === "cleanup_restore" || (action === "undo" && ["cleanup_restore", "ocr_cleanup"].includes(state.undo_action ?? ""))
          ? "blocks" : "layout",
        binding: bound, snapshot: cleanup ? next : undefined,
        clearedBlockId: action === "delete" || discardInteractions ? selection?.blockId : undefined };
      try { await finishRefresh(controller.signal, current);
        if (controller.signal.aborted || current !== generation.current) return;
        setReadingRefreshFailed(false); }
      catch { if (controller.signal.aborted || current !== generation.current) return; setReadingRefreshFailed(true); }
      if (action === "edit" || action === "delete") { onCloseSelection(); }
      else if (!cleanup && !regionAction && !regionUndo) { setAuditOffset(0); setRefresh(x => x + 1); }
      return true;
    } catch (e) { if (!controller.signal.aborted && current === generation.current) {
      if (e instanceof ReadingMutationRejected && e.status >= 400 && e.status < 500 && e.status !== 408) retry.current = null;
      setConfirmEdit(false);
      const reason = e instanceof Error ? e.message : "";
      const cleanupErrors: Record<string, string> = {
        block_manually_edited: "A related block was edited in the reader. Region changes cannot be applied.",
        block_manually_deleted: "A related block was deleted in the reader. Region changes cannot be applied.",
        cleanup_block_edited: "The block was manually edited. Reload before continuing.",
        cleanup_block_deleted: "The block was deleted. Reload before continuing.",
        cleanup_already_restored: "This record was already restored. Reload before continuing.",
        cleanup_record_missing: "This cleanup record is no longer available. Reload before continuing.",
        cleanup_mapping_unavailable: "The original content location is unavailable. Reload before continuing.",
      };
      setError(cleanupErrors[reason] ?? (reason.includes("changed") || reason.includes("conflict") || reason === "nothing_to_undo"
        ? "Content changed elsewhere. Your draft is retained. Reload before continuing."
        : "The change could not be confirmed. Retry the same action or reload; your draft is retained."));
    } }
    finally { mutating.current = false; if (current === generation.current) setBusy(false); }
  }
  const undoLabel = state?.undo_action === "ocr_cleanup" ? "Undo OCR cleanup" : "Undo last content change";
  const undoButton = <Tooltip title={undoLabel}><span><IconButton aria-label={undoLabel}
    disabled={busy || readingRefreshFailed || !state?.can_undo} onClick={() => void mutate("undo")}><UndoIcon /></IconButton></span></Tooltip>;
  const refreshNotice = readingRefreshFailed && <Alert severity="warning" action={<Button disabled={busy} onClick={async () => {
    if (mutating.current) return;
    mutating.current = true; setBusy(true);
    const controller = new AbortController(); mutationRequest.current = controller;
    const current = generation.current;
    try { await finishRefresh(controller.signal, current);
      if (!controller.signal.aborted && current === generation.current) setReadingRefreshFailed(false);
    } catch { /* The saved change is not replayed. */ }
    finally { mutating.current = false; if (current === generation.current) setBusy(false); }
  }}>Retry reading</Button>}>Changes saved. Reading refresh failed.</Alert>;
  const count = Array.from(text).length;
  return <Box className="reader-content-actions" sx={{ px: 1, py: .5 }}>
    <Stack direction="row" alignItems="center" spacing={1}>
      {state?.disposition === "needs_review" && <Alert severity="warning" sx={{ flex: 1 }}>OCR needs review</Alert>}
      <Tooltip title="Review source"><span><IconButton aria-label="Review source" onClick={() => { setAuditOffset(0); setNavigated(false); setReviewOpen(true); }} disabled={busy}><FactCheckOutlinedIcon /></IconButton></span></Tooltip>
      {!reviewOpen && !selection && undoButton}
      {error && !reviewOpen && !selection && <Alert severity="error">{error}</Alert>}
    </Stack>
    {!reviewOpen && !selection && refreshNotice}
    <Dialog open={reviewOpen || Boolean(selection)} onClose={close} disableRestoreFocus={navigated} fullWidth maxWidth="md"
      PaperProps={{ sx: !selection ? { height: "min(880px, calc(100dvh - 64px))" } : undefined }} aria-labelledby="reading-content-dialog-title">
      <DialogTitle id="reading-content-dialog-title" sx={{ display: "flex", justifyContent: "space-between" }}>
        {selection ? (selection.action === "edit" ? "Edit block" : "Delete block?") : "Source review"}
        <IconButton aria-label="Close content editor" disabled={(busy && Boolean(selection)) || mutating.current} onClick={close}><CloseIcon /></IconButton>
      </DialogTitle>
      {!selection && reviewOpen && <Stack className="source-review-tab-bar" direction="row" alignItems="center" sx={{ px: { xs: 1, sm: 3 }, flexShrink: 0 }}>
        <Tabs className="source-review-tabs" value={reviewTab} aria-label="Source review views" variant="fullWidth"
          sx={{ flex: 1, minWidth: 0 }} onChange={(_, next: "history" | "regions") => {
            if (busy || mutating.current || next === reviewTab) return;
            tabScroll.current[reviewTab] = reviewScroll.current?.scrollTop ?? 0;
            setPreview(null); setLoadedPreview(null); setConfirmRegion(null); setError(""); setReviewTab(next);
          }}>
          <Tab className="source-review-tab" id="source-review-history-tab" aria-controls="source-review-history-panel"
            value="history" label="OCR cleanup history" disabled={busy} sx={{ textTransform: "none", minWidth: 0, px: 1 }} />
          <Tab className="source-review-tab" id="source-review-regions-tab" aria-controls="source-review-regions-panel"
            value="regions" label="Source regions" disabled={busy} sx={{ textTransform: "none", minWidth: 0, px: 1 }} />
        </Tabs>
        <Tooltip title="Refresh source review"><span><IconButton aria-label="Refresh source review" disabled={busy}
          onClick={() => { if (confirmReadingNavigation()) setRefresh(value => value + 1); }}><RefreshIcon /></IconButton></span></Tooltip>
      </Stack>}
      <DialogContent ref={reviewScroll} className="source-review-dialog-content" dividers
        sx={{ overflowWrap: "anywhere", position: "relative", bgcolor: selection ? "background.paper" : "grey.50", minHeight: 0 }}>
        {busy && <CircularProgress size={22} aria-label="Loading content changes" sx={{ position: "absolute", top: 8, right: 12 }} />}
        {error && !regionEditor && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
        {refreshNotice}
        {selection?.action === "edit" && <TextField label="Block text" multiline minRows={8} maxRows={18} fullWidth value={text} disabled={!ready}
          onChange={e => setText(e.target.value)} error={count > 10000} helperText={`${count} / 10000`} />}
        {selection && <Typography className="reading-content-edit-block-id" variant="caption">{selection.blockId}</Typography>}
        {selection?.action === "delete" && <><Alert severity="warning">Delete this block and its interactions, including Notes? Generated context will be invalidated. Undo restores the block, not deleted interactions.</Alert><Typography sx={{ whiteSpace: "pre-wrap", mt: 2 }}>{text}</Typography></>}
        {!selection && reviewOpen && state && loadedKey === selectionKey && <>
          <Box className="source-review-tab-panel" id="source-review-history-panel" role="tabpanel"
            aria-labelledby="source-review-history-tab" hidden={reviewTab !== "history"}>
          <CleanupAudit key={docName} docName={docName} snapshot={state} history={state.cleanup_audit}
            active={reviewTab === "history"}
            busy={busy || readingRefreshFailed} onPage={setAuditOffset} onDirty={setCleanupDirty} onError={setError}
            onMutate={(action, command) => mutate(action, undefined, undefined, command)}
            onNavigate={async (block, snapshot, signal, assertCurrent) => {
              await onNavigate(block, snapshot, signal, assertCurrent);
              if (signal.aborted) return;
              setNavigated(true); setReviewOpen(false); setPreview(null); setLoadedPreview(null);
            }} />
          </Box>
          <Box className="source-review-tab-panel" id="source-review-regions-panel" role="tabpanel"
            aria-labelledby="source-review-regions-tab" hidden={reviewTab !== "regions"}>
          <Stack spacing={2}>
            <Typography component="h3" variant="subtitle1">Source regions ({state.regions.length})</Typography>
          {state.regions.length === 0 && <Typography>No unresolved OCR regions recorded.</Typography>}
          {state.regions.slice(regionPage * 20, (regionPage + 1) * 20).map(region => <Paper variant="outlined" key={region.id} className="source-review-region" data-region-id={region.id}
            sx={{ p: { xs: 1.5, sm: 2 }, borderRadius: 1, minWidth: 0 }}>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              <Typography>Page {region.page_index + 1} · {region.status === "unresolved" ? "Needs review" : "Ignored"}</Typography>
              <Button aria-expanded={preview === region.id} aria-controls={`source-preview-${region.id}`}
                endIcon={preview === region.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                onClick={() => { setLoadedPreview(null); setPreview(preview === region.id ? null : region.id); setError(""); }}>
                {preview === region.id ? "Hide original" : "Original"}
              </Button>
              {region.status === "unresolved" && <Button color="warning" disabled={busy || loadedPreview !== region.id} onClick={() => setConfirmRegion(region.id)}>Ignore region</Button>}
            </Stack>
            <SourceReviewManualChange region={region} />
            {region.repair?.applied_at && <Typography className="source-region-repair-applied" variant="caption" component="p">
              {region.repair.applied_text === "" ? "Region text removed" : "Region text repaired"} · {new Date(region.repair.applied_at).toLocaleString()}
            </Typography>}
            {region.raw_candidate && !region.block_ids?.length && <Typography sx={{ whiteSpace: "pre-wrap", maxHeight: 180, overflow: "auto" }}>{region.raw_candidate}</Typography>}
            <Stack direction="row" alignItems="center">
              <Button className="source-review-blocks-toggle" aria-expanded={selectedRegion === region.id}
                onClick={() => { setSelectedRegion(region.id); setRegionActivation(value => value + 1); }}>
                Related blocks ({region.block_ids?.length ?? 0})
              </Button>
              {selectedRegion === region.id && <IconButton aria-label="Collapse related blocks" onClick={() => setSelectedRegion(null)}><ExpandLessIcon /></IconButton>}
            </Stack>
            {reviewTab === "regions" && selectedRegion === region.id && <SourceReviewBlocks key={region.id} docName={docName} region={region} snapshot={state}
              activation={regionActivation} disabled={busy || readingRefreshFailed}
              onRepair={(block, mode, element) => { if (confirmReadingNavigation()) {
                regionReturnFocus.current = { element, scrollTop: reviewScroll.current?.scrollTop ?? 0 };
                setError(""); setRegionEditor({ id: region.id, block, mode });
              } }}
              onSnapshot={next => setState(current => ({ ...next, cleanup_audit: next.cleanup_audit
                ?? (current && sameReadingSnapshot(current, next) ? current.cleanup_audit : undefined) }))}
              onNavigate={async (block, snapshot, signal, assertCurrent) => {
                await onNavigate(block, snapshot, signal, assertCurrent);
                if (signal.aborted) return;
                setNavigated(true); setReviewOpen(false); setPreview(null); setLoadedPreview(null);
              }} />}
            {preview === region.id && <Box component="img" id={`source-preview-${region.id}`} alt={`Original page ${region.page_index + 1} region`} sx={{ maxWidth: "100%", maxHeight: 550, objectFit: "contain" }}
              src={`/api/documents/${encodeURIComponent(docName)}/source-review/${encodeURIComponent(region.id)}/image`}
              onLoad={() => setLoadedPreview(region.id)}
              onError={() => { setPreview(null); setLoadedPreview(null); setError("Original source is unavailable or changed. Reload before continuing."); }} />}
          </Paper>)}
          {state.regions.length > 20 && <Stack className="source-regions-pagination" direction="row" alignItems="center">
            <IconButton aria-label="Previous source regions" disabled={busy || regionPage === 0}
              onClick={() => { if (confirmReadingNavigation()) setRegionPage(value => Math.max(0, value - 1)); }}><ChevronLeftIcon /></IconButton>
            <Typography>{regionPage + 1} / {Math.ceil(state.regions.length / 20)}</Typography>
            <IconButton aria-label="Next source regions" disabled={busy || (regionPage + 1) * 20 >= state.regions.length}
              onClick={() => { if (confirmReadingNavigation()) setRegionPage(value => value + 1); }}><ChevronRightIcon /></IconButton>
          </Stack>}
          </Stack>
          </Box>
        </>}
      </DialogContent>
      <DialogActions>
        {!selection && undoButton}
        <Button onClick={close} disabled={(busy && Boolean(selection)) || mutating.current}>Cancel</Button>
        {error && <Button disabled={busy} onClick={() => { if (confirmReadingNavigation()) setRefresh(x => x + 1); }}>Reload</Button>}
        {selection?.action === "edit" && <Button startIcon={<EditOutlinedIcon />} disabled={!ready || !text.trim() || count > 10000 || readingRefreshFailed} onClick={() => setConfirmEdit(true)}>Save</Button>}
        {selection?.action === "delete" && <Button color="error" startIcon={<DeleteOutlineIcon />} disabled={!ready || readingRefreshFailed} onClick={() => void mutate("delete")}>Delete block</Button>}
      </DialogActions>
    </Dialog>
    {reviewOpen && regionEditor &&
      <SourceReviewBlockEditor key={`${docName}:${regionEditor.id}:${regionEditor.block.block_id}:${regionEditor.mode}`}
        block={regionEditor.block} regionId={regionEditor.id} mode={regionEditor.mode}
        busy={busy || readingRefreshFailed} error={error} onDirty={setRegionDirty} onClose={() => setRegionEditor(null)}
        onMutate={(action, command) => mutate(action, undefined, undefined, undefined, command)} />}
    <Dialog open={confirmEdit} onClose={() => !busy && setConfirmEdit(false)} aria-labelledby="reading-content-interactions-title">
      <DialogTitle id="reading-content-interactions-title">Keep this block's interactions?</DialogTitle>
      <DialogContent>If the text changes, translations and affected summaries are always deleted. Keep preserves other interactions, including Notes; Clear deletes them too. Identical text leaves all data unchanged. Undo does not restore deleted interactions.</DialogContent>
      <DialogActions sx={{ flexWrap: "wrap" }}>
        <Button disabled={busy} onClick={() => setConfirmEdit(false)}>Cancel</Button>
        <Button disabled={busy} onClick={() => void mutate("edit", undefined, false)}>Keep and save</Button>
        <Button disabled={busy} color="error" onClick={() => void mutate("edit", undefined, true)}>Delete interactions and save</Button>
      </DialogActions>
    </Dialog>
    <Dialog open={Boolean(confirmRegion)} onClose={() => !busy && setConfirmRegion(null)}>
      <DialogTitle>Ignore this OCR region?</DialogTitle>
      <DialogContent>This acknowledges the OCR region and removes it from pending source review. Existing body content and generation input stay unchanged. To delete content, use Delete in the body block menu. The original image is retained. This can be undone.</DialogContent>
      <DialogActions><Button disabled={busy} onClick={() => setConfirmRegion(null)}>Cancel</Button><Button disabled={busy} color="warning" onClick={() => void mutate("discard_region", confirmRegion!)}>Ignore region</Button></DialogActions>
    </Dialog>
  </Box>;
}
