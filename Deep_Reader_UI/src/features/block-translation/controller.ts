import { useLayoutEffect, useRef, useState } from "react";
import { translationService, type TranslationGenerateRequest, type TranslationReadResponse } from "../../services/TranslationService";
import type { ReadingRefresh } from "../../services/ReadingContentService";
import type { AppNotificationState } from "../../shared/components/AppNotification";
import type { ReaderContentBlock } from "../reader-content/model";
import { useInlineTranslations } from "./useInlineTranslations";
import { blockSourceIdentity, profileFrom, replaceSaved, savedTranslation, translationCommandKey, translationKey, validLanguages, targetLanguages,
  type TranslationSelection, type TranslationStatus, type TranslationSurface } from "./model";

const silent: AppNotificationState = { open: false, message: "", severity: "info" };
export function useBlockTranslation({ docName, scopeKey, blocks, service = translationService }: {
  docName: string; scopeKey: string; blocks: ReaderContentBlock[]; service?: typeof translationService;
}) {
  const [selection, setSelection] = useState<TranslationSelection | null>(null);
  const [surface, setSurface] = useState<TranslationSurface>(null);
  const [status, setStatus] = useState<TranslationStatus>("idle");
  const [data, setData] = useState<TranslationReadResponse | null>(null);
  const [from, setFrom] = useState<string | null>(null);
  const [to, setTo] = useState<string | null>(null);
  const [focusLanguage, setFocusLanguage] = useState<string | null>(null);
  const [notification, setNotification] = useState(silent);
  const selected = useRef<TranslationSelection | null>(null);
  const request = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const busy = useRef(false);
  const mounted = useRef(false);
  const returnFocus = useRef<HTMLElement | null>(null);
  // Unknown HTTP outcomes retain their idempotency identity until a saved read reconciles them.
  const uncertain = useRef(new Map<string, { command: TranslationGenerateRequest; hash: string }>());
  const previousScope = useRef(scopeKey);
  const inline = useInlineTranslations(docName, JSON.stringify([docName, scopeKey]), blocks, service);

  function cancel() { ++generation.current; request.current?.abort(); request.current = null; busy.current = false; }
  function close() {
    cancel(); selected.current = null; setSelection(null); setSurface(null); setStatus("idle");
    setData(null); setFrom(null); setTo(null); setNotification(silent); setFocusLanguage(null);
    if (returnFocus.current?.isConnected) returnFocus.current.focus({ preventScroll: true });
  }
  useLayoutEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; cancel(); };
  }, []);
  useLayoutEffect(() => {
    const current = selected.current;
    const block = blocks.find(item => item.block_id === current?.content_block_id);
    if (previousScope.current !== scopeKey || (current && (current.doc_name !== docName
      || !block || blockSourceIdentity(block) !== current.sourceIdentity))) {
      close();
    }
    previousScope.current = scopeKey;
  }, [docName, scopeKey, blocks]);

  function currentRequest(target: TranslationSelection, id: number, controller: AbortController) {
    return mounted.current && !controller.signal.aborted && generation.current === id
      && selected.current === target;
  }
  function notify(message: string, severity: AppNotificationState["severity"], actionLabel?: string) {
    setNotification({ open: true, message, severity, actionLabel });
  }
  function clearUncertain(target: TranslationSelection, language?: string) {
    for (const [key, pending] of uncertain.current) {
      if (translationKey(pending.command) === translationKey(target)
        && (!language || pending.command.target_language === language)) uncertain.current.delete(key);
    }
  }
  async function read(target: TranslationSelection, nextSurface: Exclude<TranslationSurface, null>,
    retainedFocus: string | null = null) {
    cancel();
    selected.current = target; setSelection(target); setSurface(nextSurface); setStatus("loading");
    setData(null); setFrom(null); setTo(null); setFocusLanguage(retainedFocus); setNotification(silent);
    const controller = new AbortController(); request.current = controller;
    const id = generation.current;
    try {
      const next = await service.read(target, controller.signal);
      if (!currentRequest(target, id, controller)) return;
      let unresolved: TranslationGenerateRequest | undefined;
      for (const [key, pending] of uncertain.current) {
        if (translationKey(pending.command) !== translationKey(target)) continue;
        if (pending.hash !== next.effective_content_hash || savedTranslation(next, pending.command.target_language)) {
          uncertain.current.delete(key);
        } else unresolved = pending.command;
      }
      const nextFrom = unresolved?.source_language ?? profileFrom(next);
      const nextTo = unresolved?.target_language ?? null;
      inline.reconcile(target, next.translations);
      setData(next); setFrom(nextFrom);
      setTo(nextTo === nextFrom ? null : nextTo); setStatus("ready");
      const nextFocus = savedTranslation(next, retainedFocus)?.target_language ?? null;
      setFocusLanguage(nextFocus); inline.focus(target, nextFocus);
      if (unresolved) notify("The earlier translation is unconfirmed. Retry it or choose different languages.", "warning");
    } catch {
      if (!currentRequest(target, id, controller)) return;
      setStatus("failed"); notify("Saved translations could not be loaded. Please retry.", "error");
    }
  }
  function open(block: ReaderContentBlock, mode: "translate" | "view", trigger?: HTMLElement) {
    if (!docName.trim() || !block.block_id.trim()) return;
    returnFocus.current = trigger ?? null;
    void read({ doc_name: docName, content_block_id: block.block_id, sourceIdentity: blockSourceIdentity(block) },
      mode === "translate" ? "languages" : "saved");
  }
  async function submit() {
    const target = selected.current;
    if (!target || !data || status !== "ready" || busy.current || (from && from === to)) return;
    const saved = savedTranslation(data, to);
    if (saved) {
      cancel(); clearUncertain(target, saved.target_language);
      setSurface(inline.languages.includes(saved.target_language) ? null : "saved");
      setFocusLanguage(saved.target_language); inline.focus(target, saved.target_language); setNotification(silent);
      return;
    }
    if (!validLanguages(data.language_codes, from, to)) return;
    const key = translationCommandKey(target, from!, to!), pending = uncertain.current.get(key);
    busy.current = true;
    const command = pending?.command ?? { doc_name: target.doc_name, content_block_id: target.content_block_id,
      source_language: from!, target_language: to!, mutation_id: crypto.randomUUID() };
    uncertain.current.set(key, { command, hash: data.effective_content_hash });
    request.current?.abort();
    const id = ++generation.current, controller = new AbortController(); request.current = controller;
    setSurface(null); setStatus("generating");
    notify("Translating this block...", "info", "Cancel");
    try {
      const result = await service.generate(command, controller.signal);
      if (!currentRequest(target, id, controller)) return;
      if (result.envelope.effective_content_hash !== data.effective_content_hash
        || result.envelope.reading_revision !== data.reading_revision
        || result.envelope.target.document_id !== data.target.document_id) {
        throw new Error("The source changed. Reload saved translations.");
      }
      if (!result.envelope.persisted || !result.record) {
        uncertain.current.delete(key); setStatus("failed");
        notify(result.envelope.status === "untranslatable" ? "This block could not be translated. Saved translations are unchanged."
          : "The translation was not saved. Please retry.", "warning", "Try again");
        return;
      }
      clearUncertain(target, result.record.target_language);
      const nextRecords = replaceSaved(data.translations, result.record);
      inline.reconcile(target, nextRecords);
      setData({ ...data, translations: nextRecords });
      setStatus("ready"); setSurface(inline.languages.includes(result.record.target_language) ? null : "saved");
      setFocusLanguage(result.record.target_language); inline.focus(target, result.record.target_language);
      notify(result.record.status === "partial" ? "Partial translation saved." : "Translation saved.",
        result.record.status === "partial" ? "warning" : "success");
    } catch (error) {
      if (!currentRequest(target, id, controller)) return;
      if (error instanceof Error && error.name === "TranslationRequestRejected") uncertain.current.delete(key);
      setStatus("failed");
      notify(error instanceof Error ? error.message : "Translation could not be confirmed. Please retry.", "error", "Try again");
    } finally {
      if (currentRequest(target, id, controller)) busy.current = false;
    }
  }
  function invalidate(refresh: ReadingRefresh) {
    inline.invalidate([...(refresh.translation_block_ids ?? []), ...(refresh.cleared_block_ids ?? []),
      ...(refresh.cleared_block_id ? [refresh.cleared_block_id] : [])]);
    const current = selected.current;
    if (current && (refresh.translation_block_ids?.includes(current.content_block_id)
      || refresh.cleared_block_id === current.content_block_id || refresh.cleared_block_ids?.includes(current.content_block_id))) {
      clearUncertain(current); close();
    }
  }
  return { selection, surface, status, data, from, to, focusLanguage, notification,
    getInlineTranslations: inline.recordsFor,
    getInlineTranslationState: (block: ReaderContentBlock) => ({ records: inline.recordsFor(block),
      status: inline.statusFor(block), onRetry: () => inline.retry(block),
      focusLanguage: inline.focusFor(block), canFocus: surface === null }),
    onVisibleTranslationBlocksChange: inline.observe,
    inlineLanguages: inline.languages,
    inlineLanguageOptions: [...new Set([...(data?.language_codes ?? []), ...inline.languages])].sort(),
    setInlineLanguage: (language: string, visible: boolean) => {
      if (status === "ready" && (data?.language_codes.includes(language) || inline.languages.includes(language))) {
        inline.select(language, visible);
        if (selected.current && savedTranslation(data, language) && (visible || focusLanguage === language)) {
          setFocusLanguage(language); inline.focus(selected.current, language);
        }
      }
    },
    targetLanguageCodes: targetLanguages(data?.language_codes ?? [], from),
    canSubmit: status === "ready" && Boolean(data) && from !== to
      && (Boolean(savedTranslation(data, to)) || validLanguages(data?.language_codes ?? [], from, to)),
    setFrom: (value: string | null) => { setFrom(value); setTo(current => current === value ? null : current); },
    setTo: (value: string | null) => setTo(value === from ? null : value),
    open, close, invalidate, submit,
    dismissNotification: () => setNotification(silent),
    retry: () => { if (selected.current) void read(selected.current, surface ?? "saved",
      surface === "saved" ? focusLanguage : null); },
    add: () => { if (selected.current) void read(selected.current, "languages"); },
    notificationAction: () => { if (status === "generating") close();
      else if (selected.current) void read(selected.current, "languages"); },
  };
}
export type BlockTranslationController = ReturnType<typeof useBlockTranslation>;
