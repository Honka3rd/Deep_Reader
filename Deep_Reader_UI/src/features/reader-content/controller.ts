import { useEffect, useRef, useState } from "react";
import type { DocumentTaskLayout, RequestStatus, SectionSelection } from "../../types/api";
import { assertSectionInLayout } from "../../shared/readerIdentity";
import type { TaskUnitContentService } from "../../services";
import { taskUnitContentService } from "../../services";
import { readingContentService, type ReadingRefresh, type ReadingState, type ReviewNavigationBlock } from "../../services/ReadingContentService";
import { mergeReadingRefresh, validateRefreshPage } from "./readingRefreshModel";
import { isReviewNavigationBlock, matchesReviewDisplayContent, sameReadingSnapshot } from "./sourceReviewModel";
import {
  aggregateContentBlocks,
  buildContentGroups,
  type ReaderContentBlock,
  type ReaderContentGroup,
} from "./model";

interface UseReaderContentControllerOptions {
  docName: string;
  layout: DocumentTaskLayout | null;
  service?: TaskUnitContentService;
}

export function useReaderContentController({
  docName,
  layout,
  service = taskUnitContentService,
}: UseReaderContentControllerOptions) {
  const [selectedSection, setSelectedSection] = useState<SectionSelection | null>(null);
  const [contentBlocks, setContentBlocks] = useState<ReaderContentBlock[]>([]);
  const [contentGroups, setContentGroups] = useState<ReaderContentGroup[]>([]);
  const [contentStatus, setContentStatus] = useState<RequestStatus>("initial");
  const [contentError, setContentError] = useState("");
  const [navigationTarget, setNavigationTarget] = useState<{ blockId: string; taskUnitId: string; requestId: number } | null>(null);
  const contentRequestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const refreshAbort = useRef<AbortController | null>(null);
  const readerRef = useRef({ contentGroups, contentStatus });
  readerRef.current = { contentGroups, contentStatus };
  useEffect(() => () => {
    refreshAbort.current?.abort();
    abortRef.current?.abort();
    contentRequestIdRef.current += 1;
  }, [docName, layout]);
  const scopeRef = useRef({ docName, layout });
  scopeRef.current = { docName, layout };

  function resetContent() {
    refreshAbort.current?.abort();
    abortRef.current?.abort();
    contentRequestIdRef.current += 1;
    setSelectedSection(null);
    setNavigationTarget(null);
    setContentBlocks([]);
    setContentGroups([]);
    setContentError("");
    setContentStatus("initial");
  }

  function refreshSelectionLabels(next: DocumentTaskLayout) {
    setSelectedSection(current => {
      if (!current) return current;
      const chapter = next.chapters?.find(item => item.chapter_id === current.chapter.chapter_id);
      const section = chapter?.sections?.find(item => item.section_id === current.section.section_id);
      return chapter && section ? { ...current, chapter, section } : current;
    });
  }

  async function selectSection(selection: SectionSelection, throwOnError = false, signal?: AbortSignal,
    projection: DocumentTaskLayout | null = layout,
    target?: { block: ReviewNavigationBlock; snapshot: ReadingState; assertCurrent: () => void }) {
    if (signal?.aborted) return;
    if (scopeRef.current.docName !== docName || scopeRef.current.layout !== layout) return;
    refreshAbort.current?.abort();
    abortRef.current?.abort();
    const abort = new AbortController();
    const cancel = () => abort.abort();
    signal?.addEventListener("abort", cancel, { once: true });
    abortRef.current = abort;
    const requestId = contentRequestIdRef.current + 1;
    contentRequestIdRef.current = requestId;
    setSelectedSection(selection);
    setNavigationTarget(null);
    setContentBlocks([]);
    setContentGroups([]);
    setContentError("");
    try {
      assertSectionInLayout(scopeRef.current.docName, projection, selection);
      if (scopeRef.current.docName !== docName || scopeRef.current.layout !== layout) {
        throw new Error("Document selection has changed. Select the section again.");
      }
      if (selection.taskUnits.length === 0) {
        setContentStatus("empty");
        return;
      }
      setContentStatus("loading");
      const taskUnitContents = await service.fetchTaskUnitContents(
        docName.trim(),
        selection.taskUnits.map((taskUnit) => taskUnit.unit_id),
        abort.signal,
      );
      if (abort.signal.aborted || contentRequestIdRef.current !== requestId
        || scopeRef.current.docName !== docName) {
        return;
      }
      if (taskUnitContents.some((item) =>
        (item.document_id != null && projection?.document_id != null && item.document_id !== projection.document_id)
        || (item.chapter_id != null && item.chapter_id !== selection.chapter.chapter_id)
        || (item.section_id != null && item.section_id !== selection.section.section_id))) {
        throw new Error("Content response belongs to a different document or section.");
      }
      const nextBlocks = aggregateContentBlocks(taskUnitContents);
      if (target) {
        target.assertCurrent();
        const block = target.block;
        const selected = nextBlocks.find(item => item.block_id === block.block_id
          && item.ownerContext.taskUnitId === block.task_unit_id);
        if (!selected) throw new Error("This block is no longer in the current content. Refresh source review.");
        // Content batches have no snapshot token. Revalidate the bounded target before publication.
        const latest = await readingContentService.read(docName, new URLSearchParams({ block_id: block.block_id }), abort.signal);
        if (abort.signal.aborted || contentRequestIdRef.current !== requestId
          || scopeRef.current.docName !== docName || scopeRef.current.layout !== layout) return;
        target.assertCurrent();
        const current = latest.blocks.find(item => item.block_id === block.block_id);
        if (!sameReadingSnapshot(latest, target.snapshot) || !current || !isReviewNavigationBlock(current)
          || current.chapter_id !== block.chapter_id || current.section_id !== block.section_id
          || current.task_unit_id !== block.task_unit_id || current.content !== block.content
          || !matchesReviewDisplayContent(selected.content, current.content, selected.metadata?.display_normalization)) {
          throw new Error("Content changed. Refresh source review.");
        }
      }
      setContentBlocks(nextBlocks);
      setContentGroups(buildContentGroups(taskUnitContents));
      setContentStatus(nextBlocks.length > 0 ? "success" : "empty");
      if (target) setNavigationTarget({ blockId: target.block.block_id, taskUnitId: target.block.task_unit_id, requestId });
      return true;
    } catch (error) {
      if (abort.signal.aborted) {
        if (contentRequestIdRef.current === requestId) setContentStatus("initial");
        if (throwOnError) throw error;
        return;
      }
      if (contentRequestIdRef.current !== requestId
        || scopeRef.current.docName !== docName) {
        return;
      }
      setContentStatus("error");
      // Source review owns navigation failures inside its modal focus boundary.
      setContentError(target ? "" : error instanceof Error ? error.message : String(error));
      if (throwOnError) throw error;
    } finally { signal?.removeEventListener("abort", cancel); }
  }

  async function navigateToBlock(block: ReviewNavigationBlock, snapshot: ReadingState, signal: AbortSignal,
    assertCurrent: () => void) {
    if (signal.aborted) return;
    assertCurrent();
    if (!isReviewNavigationBlock(block)) throw new Error("Content mapping unavailable.");
    const chapter = layout?.chapters?.find(item => item.chapter_id === block.chapter_id);
    const section = chapter?.sections?.find(item => item.section_id === block.section_id);
    if (!chapter || !section || !section.task_units?.some(unit => unit.unit_id === block.task_unit_id)) {
      throw new Error("This content belongs to an outdated structure. Reload the document.");
    }
    const selected = await selectSection({ chapter, section, taskUnits: section.task_units }, true, signal, layout,
      { block, snapshot, assertCurrent });
    if (!selected) throw new Error("Content selection changed. Select the related block again.");
  }

  async function refreshFromLayout(next: DocumentTaskLayout, signal?: AbortSignal) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const chapter = next.chapters?.find(item => item.chapter_id === selectedSection?.chapter.chapter_id);
    const section = chapter?.sections?.find(item => item.section_id === selectedSection?.section.section_id);
    if (!chapter || !section) { resetContent(); return; }
    // Read against the freshly fetched projection before App publishes it. This
    // retains stable selection and cannot send old unit IDs after structural change.
    await selectSection({ chapter, section, taskUnits: section.task_units || [] }, true, signal, next);
  }

  async function refreshAffectedBlocks(binding: ReadingRefresh, signal: AbortSignal) {
    refreshAbort.current?.abort();
    const abort = new AbortController(); refreshAbort.current = abort;
    const cancel = () => abort.abort();
    signal.addEventListener("abort", cancel, { once: true });
    const requestId = contentRequestIdRef.current;
    const assertCurrent = () => {
      if (signal.aborted || abort.signal.aborted || scopeRef.current.docName !== docName
        || scopeRef.current.layout !== layout || requestId !== contentRequestIdRef.current) throw new DOMException("Aborted", "AbortError");
    };
    try {
      assertCurrent();
      if (!layout || layout.document_id !== binding.document_id || readerRef.current.contentStatus === "loading") {
        throw new Error("Reading selection changed. Retry reading refresh.");
      }
      validateRefreshPage(binding, binding, 0);
      const changes: ReadingRefresh["blocks"] = [];
      // Reread page zero even on retry, so a committed receipt never substitutes for current state.
      let offset = 0;
      do {
        const page = await readingContentService.readRefresh(docName, binding, offset, abort.signal);
        assertCurrent(); validateRefreshPage(page, binding, offset);
        changes.push(...page.blocks); offset += 20;
      } while (offset < binding.block_page.total);
      assertCurrent();
      const groups = mergeReadingRefresh(readerRef.current.contentGroups, layout, binding, changes);
      setContentGroups(groups); setContentBlocks(groups.flatMap(group => group.blocks));
      if (groups.length) setContentStatus(groups.some(group => group.blocks.length) ? "success" : "empty");
    } finally { signal.removeEventListener("abort", cancel); }
  }

  return {
    selectedSection,
    contentBlocks,
    contentGroups,
    contentStatus,
    contentError,
    navigationTarget,
    navigateToBlock,
    selectSection,
    retryContent: () => selectedSection ? selectSection(selectedSection) : Promise.resolve(),
    refreshChangedContent: (signal?: AbortSignal) => selectedSection ? selectSection(selectedSection, true, signal) : Promise.resolve(),
    resetContent,
    refreshSelectionLabels,
    refreshFromLayout,
    refreshAffectedBlocks,
  };
}
