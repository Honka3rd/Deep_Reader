import { useEffect, useRef, useState } from "react";
import { readingPreferencesService, type BookPreferences } from "../../services/ReadingPreferencesService";

type ReadMode = (doc: string, signal: AbortSignal) => Promise<BookPreferences>;

// One queue per Library owner; rows own cancellation, including queued reads.
export function createReadingModeQueue(): ReadMode {
  let running = 0;
  const queued: { start: () => void; cancel: () => void }[] = [];
  function pump() {
    while (running < 2 && queued.length) queued.shift()!.start();
  }
  return (doc, signal) => new Promise((resolve, reject) => {
    const job = {
      cancel: () => {
        const index = queued.indexOf(job);
        if (index >= 0) queued.splice(index, 1);
        signal.removeEventListener("abort", job.cancel);
        reject(new DOMException("Aborted", "AbortError"));
      },
      start: () => {
        signal.removeEventListener("abort", job.cancel);
        if (signal.aborted) { job.cancel(); return; }
        running++;
        void readingPreferencesService.book(doc, signal).then(resolve, reject).finally(() => { running--; pump(); });
      },
    };
    if (signal.aborted) { job.cancel(); return; }
    signal.addEventListener("abort", job.cancel, { once: true });
    queued.push(job); pump();
  });
}

type LabelState = { identity: string; status: "loading" | "ready" | "error"; book?: BookPreferences };
export function useReadingModeLabel(doc: string, active: boolean, read: ReadMode) {
  const element = useRef<HTMLDivElement>(null);
  const [attempt, setAttempt] = useState(0);
  const identity = JSON.stringify([doc, active, attempt]);
  const current = useRef(identity); current.current = identity;
  const [state, setState] = useState<LabelState>({ identity: "", status: "loading" });
  useEffect(() => {
    setState({ identity, status: "loading" });
    if (!active || !doc.trim() || !element.current) return;
    let disposed = false, settled = false, request: AbortController | null = null;
    const observer = new IntersectionObserver(entries => {
      if (disposed || settled) return;
      if (!entries.some(entry => entry.isIntersecting)) {
        request?.abort(); request = null;
        return;
      }
      if (request) return;
      const work = new AbortController(); request = work;
      const valid = () => !disposed && !work.signal.aborted && current.current === identity && request === work;
      void read(doc, work.signal).then(book => {
        if (valid()) { settled = true; setState({ identity, status: "ready", book }); }
      }).catch(() => {
        if (valid()) { settled = true; setState({ identity, status: "error" }); }
      });
    });
    observer.observe(element.current);
    return () => { disposed = true; observer.disconnect(); request?.abort(); };
  }, [identity, doc, active, read]);
  return { element, state: state.identity === identity ? state : { identity, status: "loading" as const },
    retry: () => setAttempt(value => value + 1) };
}
