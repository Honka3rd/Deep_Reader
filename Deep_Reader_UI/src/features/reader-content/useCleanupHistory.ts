import { useEffect, useRef, useState } from "react";
import { readingContentService, type CleanupLink, type CleanupRecord, type ReadingState } from "../../services/ReadingContentService";
import { isReviewNavigationBlock, sameReadingSnapshot } from "./sourceReviewModel";
import type { ReviewNavigate } from "./useSourceReviewBlocks";

export type CleanupCommand = { cleanup_record_id: string; text?: string };
export type CleanupAction = "cleanup_edit" | "cleanup_restore" | "cleanup_delete";

export function useCleanupHistory({ docName, snapshot, onMutate, onNavigate, onDirty, onError, active = true, blocked = false }: {
  active?: boolean; blocked?: boolean;
  docName: string; snapshot: ReadingState;
  onMutate: (action: CleanupAction, command: CleanupCommand) => Promise<boolean | undefined>;
  onNavigate: ReviewNavigate; onDirty: (dirty: boolean) => void; onError: (message: string) => void;
}) {
  const [editor, setEditor] = useState<{ id: string; text: string; saved: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const request = useRef<AbortController | null>(null);
  const submitFocus = useRef<{ focused: Element | null; row: HTMLElement | null;
    scroll: HTMLElement | null | undefined; top: number | undefined; generation: number } | null>(null);
  const deleteFocus = useRef<{ target: Element | null | undefined; scroll: HTMLElement | null | undefined; top: number | undefined } | null>(null);
  const generation = useRef(0), locked = useRef(false);
  const dirty = editor !== null && editor.text !== editor.saved;
  useEffect(() => { onDirty(dirty); }, [dirty, onDirty]);
  useEffect(() => {
    if (active) return;
    ++generation.current; request.current?.abort(); locked.current = false; setBusy(false);
    submitFocus.current = null;
  }, [active]);
  useEffect(() => () => { ++generation.current; request.current?.abort();
    submitFocus.current = null;
    deleteFocus.current = null; locked.current = false; onDirty(false); }, [docName, onDirty]);
  useEffect(() => {
    if (busy || blocked || !active) return;
    const saved = submitFocus.current; submitFocus.current = null;
    if (!saved || saved.generation !== generation.current) return;
    // Restore only after both owners have re-enabled the record controls.
    const target = saved.focused instanceof HTMLElement && saved.focused.isConnected
      && saved.row?.contains(saved.focused) && !saved.focused.matches(":disabled")
      ? saved.focused : saved.row?.querySelector<HTMLElement>("button:not(:disabled)");
    target?.focus({ preventScroll: true });
    if (saved.scroll?.isConnected && saved.top !== undefined) saved.scroll.scrollTop = saved.top;
  }, [busy, blocked, active]);
  function finishDeleteFocus() {
    const saved = deleteFocus.current; deleteFocus.current = null;
    if (!saved) return;
    const scope = saved.target?.isConnected ? saved.target : document.querySelector(".cleanup-history");
    const target = scope?.querySelector<HTMLElement>("button:not(:disabled)") ?? scope;
    if (target instanceof HTMLElement) target.focus({ preventScroll: true });
    if (saved.scroll?.isConnected && saved.top !== undefined) saved.scroll.scrollTop = saved.top;
  }
  async function submit(row: CleanupRecord, action: CleanupAction) {
    if (!active || !row.record_id || locked.current) return;
    locked.current = true; setBusy(true);
    const focused = document.activeElement;
    const rowElement = document.querySelector<HTMLElement>(`.cleanup-history-record[data-record-id="${row.record_id}"]`);
    const scroll = rowElement?.closest<HTMLElement>(".source-review-dialog-content");
    const scrollTop = scroll?.scrollTop;
    const neighbour = rowElement?.nextElementSibling?.matches(".cleanup-history-record")
      ? rowElement.nextElementSibling : rowElement?.previousElementSibling;
    const focusAfterDelete = neighbour?.matches(".cleanup-history-record") ? neighbour : rowElement?.closest(".cleanup-history");
    const current = generation.current;
    try {
      const text = action !== "cleanup_delete" && editor?.id === row.record_id ? editor.text : undefined;
      const success = await onMutate(action, { cleanup_record_id: row.record_id, ...(text !== undefined ? { text } : {}) });
      if (success) {
        if (current === generation.current) {
          setEditor(null); onDirty(false);
          if (action === "cleanup_delete") deleteFocus.current = { target: focusAfterDelete, scroll, top: scrollTop };
        }
      }
      return current === generation.current ? Boolean(success) : undefined;
    } finally { if (current === generation.current) {
      locked.current = false; setBusy(false);
      // Delete confirmation owns focus until its exit transition finishes.
      if (action !== "cleanup_delete") submitFocus.current = {
        focused, row: rowElement, scroll, top: scrollTop, generation: current,
      };
    } }
  }
  async function navigate(link: CleanupLink) {
    if (!active || locked.current) return;
    locked.current = true; setBusy(true); onError("");
    const current = ++generation.current;
    const controller = new AbortController(); request.current = controller;
    const assertCurrent = () => { if (controller.signal.aborted || generation.current !== current) throw new DOMException("Aborted", "AbortError"); };
    try {
      const next = await readingContentService.read(docName, new URLSearchParams({ block_id: link.block_id }), controller.signal);
      assertCurrent();
      const block = next.blocks.find(item => item.block_id === link.block_id);
      if (!sameReadingSnapshot(snapshot, next) || !block || !isReviewNavigationBlock(block)
        || block.task_unit_id !== link.task_unit_id || block.section_id !== link.section_id || block.chapter_id !== link.chapter_id) {
        throw new Error("Content changed. Reload cleanup history before navigating.");
      }
      await onNavigate(block, next, controller.signal, assertCurrent);
    } catch (error) {
      if (!controller.signal.aborted && current === generation.current) onError(error instanceof Error ? error.message : "Navigation failed.");
    } finally { if (current === generation.current) { locked.current = false; setBusy(false); } }
  }
  return { editor, setEditor, dirty, busy, submit, navigate, finishDeleteFocus };
}
