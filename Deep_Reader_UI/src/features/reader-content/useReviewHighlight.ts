import { useEffect, useRef, useState, type RefObject } from "react";
import type { ReaderContentGroup } from "./model";

export type ReviewNavigationTarget = { blockId: string; taskUnitId: string; requestId: number };

export function useReviewHighlight(target: ReviewNavigationTarget | null | undefined,
  groups: ReaderContentGroup[], ready: boolean, container: RefObject<HTMLElement>) {
  const previousTarget = useRef<typeof target>(null);
  const [selection, setSelection] = useState<{ target: ReviewNavigationTarget; groups: ReaderContentGroup[] } | null>(null);

  useEffect(() => {
    // Only a new verified navigation may activate a highlight. Refreshes cannot revive it.
    if (previousTarget.current !== target) {
      previousTarget.current = target;
      setSelection(target && ready ? { target, groups } : null);
    } else {
      setSelection(current => current && ready && current.groups === groups ? current : null);
    }
  }, [target, groups, ready]);

  const active = ready && selection && selection.target === target && selection.groups === groups ? target : null;
  useEffect(() => {
    if (!active) return;
    const dismiss = (event: Event) => {
      const mark = event.target instanceof Element ? event.target.closest(".reader-content-navigation-highlight") : null;
      if (!mark || !container.current?.contains(mark)) setSelection(null);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setSelection(null); };
    document.addEventListener("click", dismiss, true);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("click", dismiss, true);
      document.removeEventListener("keydown", escape);
    };
  }, [active, container]);
  return active;
}
