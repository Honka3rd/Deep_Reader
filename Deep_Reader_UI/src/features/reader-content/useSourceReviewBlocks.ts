import { useEffect, useRef, useState } from "react";
import { readingContentService, type ReadingState, type ReviewNavigationBlock } from "../../services/ReadingContentService";
import { canNavigateReviewBlock, isReviewNavigationBlock } from "./sourceReviewModel";

export type ReviewNavigate = (block: ReviewNavigationBlock, snapshot: ReadingState, signal: AbortSignal,
  assertCurrent: () => void) => Promise<void>;

export function useSourceReviewBlocks(docName: string, regionId: string, activation: number,
  snapshot: ReadingState, onSnapshot: (next: ReadingState) => void, onNavigate: ReviewNavigate) {
  const [offset, setOffset] = useState(0);
  const [state, setState] = useState<ReadingState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const key = JSON.stringify([docName, regionId, activation, offset]);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const generation = useRef(0);
  const request = useRef<AbortController | null>(null);
  const locked = useRef(false);
  const latest = useRef({ docName, regionId, snapshot, onSnapshot, onNavigate });
  latest.current = { docName, regionId, snapshot, onSnapshot, onNavigate };

  async function read(blockId?: string) {
    if (locked.current) return;
    locked.current = true;
    const owner = latest.current;
    const current = ++generation.current;
    const controller = new AbortController();
    request.current = controller;
    const valid = () => !controller.signal.aborted && current === generation.current
      && latest.current.docName === owner.docName && latest.current.regionId === owner.regionId;
    const assertCurrentSnapshot = (next: ReadingState) => {
      for (const expected of [owner.snapshot, latest.current.snapshot]) {
        if (next.source_token !== expected.source_token || next.revision < expected.revision
          || (next.structure_version ?? null) !== (expected.structure_version ?? null)) {
          throw new Error("Content changed. Refresh source review.");
        }
      }
    };
    setError(""); setBusy(true);
    let accepted = false;
    try {
      // Strict Mode cleanup cancels before transport dispatch.
      await Promise.resolve();
      if (!valid()) return;
      const next = await readingContentService.read(owner.docName,
        new URLSearchParams({ region_id: owner.regionId, block_offset: String(offset) }), controller.signal);
      if (!valid()) return;
      assertCurrentSnapshot(next);
      const region = next.regions.find(item => item.id === owner.regionId);
      if (!region) throw new Error("This region changed. Refresh source review.");
      const limit = next.block_page?.limit || 20;
      const total = next.block_page?.total ?? next.blocks.length;
      const lastOffset = Math.max(0, Math.floor((total - 1) / limit) * limit);
      if (offset > lastOffset) { setOffset(lastOffset); return; }
      latest.current.onSnapshot(next);
      if (!valid()) return;
      assertCurrentSnapshot(next);
      setState(next);
      accepted = true;
      if (blockId) {
        const block = next.blocks.find(item => item.block_id === blockId);
        if (!block || !isReviewNavigationBlock(block) || !canNavigateReviewBlock(region, block)) {
          throw new Error("This content is no longer available. Select a current related block.");
        }
        await latest.current.onNavigate(block, next, controller.signal, () => {
          if (!valid()) throw new DOMException("Aborted", "AbortError");
          assertCurrentSnapshot(next);
        });
      }
    } catch (cause) {
      if (valid()) {
        if (!accepted) setState(null);
        setError(cause instanceof Error ? cause.message : "Unable to load region blocks.");
      }
    } finally {
      if (valid()) { locked.current = false; setLoadedKey(key); setBusy(false); }
    }
  }

  useEffect(() => {
    void read();
    return () => { request.current?.abort(); ++generation.current; locked.current = false; };
  }, [docName, regionId, activation, offset]);

  useEffect(() => {
    if (!state || snapshot.revision <= state.revision || snapshot.source_token !== state.source_token) return;
    const binding = snapshot.refresh;
    const region = snapshot.regions.find(row => row.id === regionId);
    if (!binding?.scope_current || binding.revision !== snapshot.revision || !region) return;
    request.current?.abort(); ++generation.current; locked.current = false;
    const total = (region.block_ids?.length ?? 0) - (region.related_deleted_count ?? 0);
    const lastOffset = Math.max(0, Math.floor((total - 1) / 20) * 20);
    if (offset > lastOffset) { setOffset(lastOffset); return; }
    const blocks = new Map(state.blocks.filter(row => region.block_ids?.includes(row.block_id)).map(row => [row.block_id, row]));
    for (const row of binding.blocks) {
      if (!region.block_ids?.includes(row.block_id)) continue;
      if (row.deleted) blocks.delete(row.block_id);
      else if (blocks.has(row.block_id) || total <= 20) blocks.set(row.block_id, { ...blocks.get(row.block_id), block_id: row.block_id, task_unit_id: row.task_unit_id,
        section_id: row.section_id, chapter_id: row.chapter_id, content: row.content_block!.content,
        deleted: false, modified: false, manually_edited: row.manually_edited });
    }
    for (const row of snapshot.blocks) if (blocks.has(row.block_id) && !row.deleted) blocks.set(row.block_id, row);
    const expectedPageSize = Math.min(20, Math.max(0, total - offset));
    if (binding.block_page.total > binding.blocks.length || blocks.size < expectedPageSize) {
      void read(); return;
    }
    const ordered = (region.block_ids ?? []).flatMap(id => blocks.has(id) ? [blocks.get(id)!] : []);
    setState({ ...snapshot, blocks: ordered.slice(0, 20), block_page: { total, offset, limit: 20 } });
    setLoadedKey(key); setBusy(false); setError("");
  }, [snapshot.revision, snapshot.refresh, state, regionId, offset, key]);

  return { state: loadedKey === key ? state : null, error: loadedKey === key ? error : "",
    busy: busy || loadedKey !== key, offset, setOffset, reload: () => void read(), navigate: (blockId: string) => void read(blockId) };
}
