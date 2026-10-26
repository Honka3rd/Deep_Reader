import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

const minimumWidth = 300;
const contentMinimumWidth = 320;
const separatorWidth = 16;

export function useReaderPaneResize() {
  const workspaceRef = useRef<HTMLElement | null>(null);
  const drag = useRef<{ element: HTMLElement; pointerId: number; x: number; width: number } | null>(null);
  const [preferredWidth, setPreferredWidth] = useState<number | null>(null);
  const [bounds, setBounds] = useState({ available: 0, narrow: true });
  const [dragging, setDragging] = useState(false);
  const maximumWidth = Math.max(minimumWidth, bounds.available - separatorWidth - contentMinimumWidth);
  const clamp = (value: number) => Math.max(minimumWidth, Math.min(maximumWidth, value));
  const width = clamp(preferredWidth ?? bounds.available * 0.36);

  function releaseCapture() {
    const current = drag.current;
    drag.current = null;
    if (current?.element.hasPointerCapture(current.pointerId)) current.element.releasePointerCapture(current.pointerId);
  }

  function stopDragging() {
    releaseCapture();
    setDragging(false);
  }

  useEffect(() => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    const narrow = window.matchMedia("(max-width: 840px)");
    const measure = () => {
      const style = getComputedStyle(workspace);
      setBounds({ available: workspace.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight), narrow: narrow.matches });
      releaseCapture();
      setDragging(false);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(workspace);
    narrow.addEventListener("change", measure);
    return () => {
      observer.disconnect();
      narrow.removeEventListener("change", measure);
      releaseCapture();
    };
  }, []);

  function onPointerDown(event: PointerEvent<HTMLElement>) {
    if (bounds.narrow || event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { element: event.currentTarget, pointerId: event.pointerId, x: event.clientX, width };
    setDragging(true);
  }

  function onPointerMove(event: PointerEvent<HTMLElement>) {
    const current = drag.current;
    if (current?.pointerId === event.pointerId) setPreferredWidth(clamp(current.width + event.clientX - current.x));
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (bounds.narrow) return;
    const step = event.shiftKey ? 48 : 16;
    const next = { ArrowLeft: width - step, ArrowRight: width + step, Home: minimumWidth, End: maximumWidth }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    stopDragging();
    setPreferredWidth(clamp(next));
  }

  return { workspaceRef, width: bounds.available ? width : null, minimumWidth, maximumWidth, dragging,
    separatorProps: {
      onPointerDown, onPointerMove, onPointerUp: stopDragging, onPointerCancel: stopDragging,
      onLostPointerCapture: stopDragging, onKeyDown,
      onDoubleClick: () => { stopDragging(); setPreferredWidth(null); },
    } };
}
