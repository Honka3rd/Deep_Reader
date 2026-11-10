import { useEffect, useRef, useState } from "react";
import { readingPreferencesService, type BookPreferences } from "../../services/ReadingPreferencesService";
import { pageSources, emptyPage, type LearningSessionMemory, type PageSource } from "./model";
import { LearningPageSession } from "./session";

export function useLearningPage(book: BookPreferences | null, sources: PageSource[], memory?: LearningSessionMemory) {
  const identity = JSON.stringify(book ? [book.doc_name, book.revision, book.workspace.revision, book.source_language, book.workspace.native_language, book.saved_settings, sources] : null);
  const current = useRef(identity); current.current = identity;
  const owner = useRef<LearningPageSession | null>(null);
  const [snapshot, setSnapshot] = useState({ identity, state: emptyPage() });
  const [reload, setReload] = useState(0);
  useEffect(() => {
    setSnapshot({ identity, state: emptyPage() });
    if (!book || !memory) return;
    const session = new LearningPageSession(book, sources, memory, state => {
      if (current.current === identity && owner.current === session && !session.signal.aborted) setSnapshot({ identity, state });
    });
    owner.current = session;
    // StrictMode's probe unmounts before this read starts; effects never directly dispatch a paid call.
    void Promise.resolve().then(() => session.start());
    return () => { session.stop(); if (owner.current === session) owner.current = null; };
    // identity includes all source/settings fields; array allocation alone is not a new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity, memory, reload]);
  return { identity, state: snapshot.identity === identity ? snapshot.state : emptyPage(),
    reload: () => setReload(value => value + 1),
    check: (key: string) => owner.current?.enqueue(key, "check"),
    refresh: (key: string) => owner.current?.enqueue(key, "refresh"),
    retry: (key: string, acknowledgePossibleDuplicateCost = false) => owner.current?.enqueue(key, "retry", acknowledgePossibleDuplicateCost),
  };
}

export function useReaderMode(scope: { docName: string; suspended: boolean; revision: number }, blocks: import("../reader-content/model").ReaderContentBlock[]) {
  let sources: PageSource[] = [], invalid = false;
  try { sources = pageSources(scope.docName, blocks); } catch { invalid = true; }
  const identity = JSON.stringify([scope.docName, scope.revision, sources, scope.suspended, invalid]);
  const active = useRef(identity); active.current = identity;
  const [state, setState] = useState<{ identity: string; book?: BookPreferences; error?: string }>({ identity: "" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    // Retire the previous confirmation even while suspended. Returning to the
    // same page must read the newly applied mode before mounting paid work.
    setState({ identity });
    if (scope.suspended || invalid || !scope.docName || !sources.length) return;
    const abort = new AbortController();
    void Promise.resolve().then(async () => {
      if (abort.signal.aborted) return;
      try {
        const book = await readingPreferencesService.book(scope.docName, abort.signal);
        if (!abort.signal.aborted && active.current === identity) setState({ identity, book });
      } catch { if (!abort.signal.aborted && active.current === identity) setState({ identity, error: "Reading mode could not be checked. Original text remains available." }); }
    });
    return () => abort.abort();
    // identity captures all scope/source values; normal renders must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity, attempt]);
  const current = state.identity === identity ? state : { identity };
  return { current, sources, identity, hidden: scope.suspended || !scope.docName || !sources.length || invalid,
    retryRead: () => setAttempt(value => value + 1) };
}

export function useOrdinaryReading(ordinary: boolean, blocks: import("../reader-content/model").ReaderContentBlock[],
  onVisibleBlocks?: (blocks: import("../reader-content/model").ReaderContentBlock[]) => void) {
  const visibleKey = JSON.stringify(blocks.map(b => [b.block_id, b.content, b.ownerContext.sourceHash]));
  useEffect(() => {
    onVisibleBlocks?.(ordinary ? blocks : []);
    return () => onVisibleBlocks?.([]);
    // Source identity, not transient array allocation, owns ordinary saved reads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ordinary, visibleKey, onVisibleBlocks]);
}
