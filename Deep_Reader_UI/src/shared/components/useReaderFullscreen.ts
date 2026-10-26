import { useEffect, useLayoutEffect, useRef, useState } from "react";

export function useReaderFullscreen(scopeKey: string) {
  const [active, setActive] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const session = useRef({ active: false, native: false, generation: 0 });

  function releaseNative() {
    if (session.current.native && document.fullscreenElement === document.documentElement) {
      void document.exitFullscreen().catch(() => {});
    }
    session.current.native = false;
  }

  function exit() {
    session.current.active = false;
    ++session.current.generation;
    releaseNative();
    setActive(false);
    buttonRef.current?.focus({ preventScroll: true });
  }

  function toggle() {
    if (session.current.active) { exit(); return; }
    session.current.active = true;
    const generation = ++session.current.generation;
    setActive(true);
    // Fullscreen the root so existing body-portalled dialogs/drawers remain visible.
    const root = document.documentElement;
    if (!document.fullscreenEnabled || !root.requestFullscreen || document.fullscreenElement) return;
    session.current.native = true;
    void root.requestFullscreen().then(() => {
      if (!session.current.active && document.fullscreenElement === root) {
        void document.exitFullscreen().catch(() => {});
      }
    }).catch(() => {
      if (generation === session.current.generation) session.current.native = false;
      // Viewport reading mode remains usable when native fullscreen is unavailable.
    });
  }

  useLayoutEffect(() => {
    document.documentElement.classList.toggle("reader-fullscreen", active);
    return () => document.documentElement.classList.remove("reader-fullscreen");
  }, [active]);

  useEffect(() => {
    session.current.active = false;
    setActive(false);
    const change = () => {
      if (session.current.native && document.fullscreenElement !== document.documentElement) exit();
    };
    const key = (event: KeyboardEvent) => {
      if (!session.current.active || event.key !== "Escape" || event.defaultPrevented) return;
      const overlay = Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"], [role="menu"]'))
        .some(element => element.getClientRects().length > 0);
      if (overlay) return;
      event.preventDefault();
      exit();
    };
    document.addEventListener("fullscreenchange", change);
    document.addEventListener("keydown", key);
    return () => {
      session.current.active = false;
      ++session.current.generation;
      document.removeEventListener("fullscreenchange", change);
      document.removeEventListener("keydown", key);
      releaseNative();
    };
  }, [scopeKey]);

  return { active, toggle, buttonRef };
}
