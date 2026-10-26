import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReaderContentBlock } from "../reader-content/model";
import type { TranslationSavedRecord, translationService } from "../../services/TranslationService";
import { blockSourceIdentity, savedSections, type TranslationSelection } from "./model";

export type InlineTranslationStatus = "idle" | "loading" | "ready" | "failed";
interface Snapshot {
  scope: string;
  sourceIdentity: string;
  records: TranslationSavedRecord[];
  status: InlineTranslationStatus;
}

// Preferences live with App; source-bound snapshots live only with the loaded scope.
export function useInlineTranslations(docName: string, scope: string, blocks: ReaderContentBlock[], service: typeof translationService) {
  const [languages, setLanguages] = useState<string[]>([]);
  const [focus, setFocus] = useState<{ scope: string; target: TranslationSelection; language: string } | null>(null);
  const [visible, setVisible] = useState<{ scope: string; blocks: ReaderContentBlock[] }>({ scope, blocks: [] });
  const [, render] = useState(0);
  const [retryVersion, setRetryVersion] = useState(0);
  const live = useRef<{ scope: string; identities: Map<string, string> }>({ scope, identities: new Map() });
  const cache = useRef(new Map<string, Snapshot>());
  const requests = useRef(new Map<string, AbortController>());
  const enabled = languages.length > 0;
  const visibleKey = JSON.stringify(visible.blocks.map(block => [block.block_id, blockSourceIdentity(block)]));
  const notify = () => render(value => value + 1);

  useLayoutEffect(() => {
    const identities = new Map(blocks.map(block => [block.block_id, blockSourceIdentity(block)]));
    live.current = { scope, identities };
    let changed = false;
    for (const [id, entry] of cache.current) {
      if (entry.scope !== scope || identities.get(id) !== entry.sourceIdentity) {
        requests.current.get(id)?.abort(); requests.current.delete(id);
        cache.current.delete(id); changed = true;
      }
    }
    if (changed) notify();
    setFocus(current => current && (current.scope !== scope
      || identities.get(current.target.content_block_id) !== current.target.sourceIdentity) ? null : current);
  }, [scope, blocks]);

  function snapshot(block: ReaderContentBlock) {
    const entry = cache.current.get(block.block_id);
    return entry?.scope === scope && entry.sourceIdentity === blockSourceIdentity(block) ? entry : undefined;
  }
  function reconcile(target: TranslationSelection, records: TranslationSavedRecord[]) {
    requests.current.get(target.content_block_id)?.abort(); requests.current.delete(target.content_block_id);
    cache.current.set(target.content_block_id, { scope, sourceIdentity: target.sourceIdentity,
      records: savedSections(records), status: "ready" });
    setFocus(current => current?.target.content_block_id === target.content_block_id
      && !records.some(record => record.target_language === current.language) ? null : current);
    notify();
  }

  useEffect(() => {
    if (!enabled || visible.scope !== scope || !docName) return;
    let active = true;
    const owned = new Map<string, AbortController>();
    const queue = visible.blocks.filter(block => blocks.some(current => current.block_id === block.block_id
      && blockSourceIdentity(current) === blockSourceIdentity(block)) && !snapshot(block));
    async function worker() {
      while (active && queue.length) {
        const block = queue.shift()!;
        // Explicit drawer reads/generation may have supplied a queued snapshot already.
        if (snapshot(block) || live.current.scope !== scope
          || live.current.identities.get(block.block_id) !== blockSourceIdentity(block)) continue;
        const controller = new AbortController(), id = block.block_id;
        const identity = blockSourceIdentity(block);
        owned.set(id, controller); requests.current.set(id, controller);
        cache.current.set(id, { scope, sourceIdentity: identity, records: [], status: "loading" }); notify();
        const current = () => active && !controller.signal.aborted && requests.current.get(id) === controller
          && live.current.scope === scope && live.current.identities.get(id) === identity;
        try {
          const result = await service.read({ doc_name: docName, content_block_id: id }, controller.signal);
          if (!current()) continue;
          cache.current.set(id, { scope, sourceIdentity: identity, records: savedSections(result.translations), status: "ready" });
        } catch {
          if (!current()) continue;
          cache.current.set(id, { scope, sourceIdentity: identity, records: [], status: "failed" });
        } finally {
          if (current()) { requests.current.delete(id); notify(); }
        }
      }
    }
    void worker(); void worker();
    return () => {
      active = false;
      for (const [id, controller] of owned) {
        controller.abort();
        if (requests.current.get(id) === controller) {
          requests.current.delete(id);
          if (cache.current.get(id)?.status === "loading") cache.current.delete(id);
        }
      }
    };
    // Snapshot publication must not restart the queue or cancel sibling reads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docName, scope, visible.scope, visibleKey, enabled, service, retryVersion]);

  const observe = useCallback((next: ReaderContentBlock[]) => setVisible({ scope, blocks: next }), [scope]);
  return {
    languages, observe, reconcile,
    focus: (target: TranslationSelection, language: string | null) => setFocus(language ? { scope, target, language } : null),
    focusFor: (block: ReaderContentBlock) => focus?.scope === scope
      && focus.target.content_block_id === block.block_id && focus.target.sourceIdentity === blockSourceIdentity(block)
      && snapshot(block)?.records.some(record => record.target_language === focus.language) ? focus.language : null,
    select: (language: string, selected: boolean) => setLanguages(current => selected
      ? [...new Set([...current, language])].sort() : current.filter(code => code !== language)),
    recordsFor: (block: ReaderContentBlock) => snapshot(block)?.records.filter(record => languages.includes(record.target_language)) ?? [],
    statusFor: (block: ReaderContentBlock): InlineTranslationStatus => enabled ? snapshot(block)?.status ?? "idle" : "idle",
    retry: (block: ReaderContentBlock) => {
      if (snapshot(block)?.status !== "failed") return;
      cache.current.delete(block.block_id); setRetryVersion(value => value + 1);
    },
    invalidate: (ids: string[]) => {
      for (const id of ids) {
        requests.current.get(id)?.abort(); requests.current.delete(id);
        const block = blocks.find(item => item.block_id === id);
        if (block) cache.current.set(id, { scope, sourceIdentity: blockSourceIdentity(block), records: [], status: "ready" });
        else cache.current.delete(id);
      }
      setFocus(current => current && ids.includes(current.target.content_block_id) ? null : current);
      if (ids.length) notify();
    },
  };
}
