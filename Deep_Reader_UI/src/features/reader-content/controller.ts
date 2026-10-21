import { useEffect, useRef, useState } from "react";
import type { DocumentTaskLayout, RequestStatus, SectionSelection } from "../../types/api";
import { assertSectionInLayout } from "../../shared/readerIdentity";
import type { TaskUnitContentService } from "../../services";
import { taskUnitContentService } from "../../services";
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
  const contentRequestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => {
    abortRef.current?.abort();
    contentRequestIdRef.current += 1;
  }, [docName, layout]);
  const scopeRef = useRef({ docName, layout });
  scopeRef.current = { docName, layout };

  function resetContent() {
    abortRef.current?.abort();
    contentRequestIdRef.current += 1;
    setSelectedSection(null);
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

  async function selectSection(selection: SectionSelection) {
    if (scopeRef.current.docName !== docName || scopeRef.current.layout !== layout) return;
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;
    const requestId = contentRequestIdRef.current + 1;
    contentRequestIdRef.current = requestId;
    setSelectedSection(selection);
    setContentBlocks([]);
    setContentGroups([]);
    setContentError("");
    try {
      assertSectionInLayout(scopeRef.current.docName, scopeRef.current.layout, selection);
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
      if (contentRequestIdRef.current !== requestId
        || scopeRef.current.docName !== docName) {
        return;
      }
      if (taskUnitContents.some((item) =>
        (item.document_id != null && layout?.document_id != null && item.document_id !== layout.document_id)
        || (item.chapter_id != null && item.chapter_id !== selection.chapter.chapter_id)
        || (item.section_id != null && item.section_id !== selection.section.section_id))) {
        throw new Error("Content response belongs to a different document or section.");
      }
      const nextBlocks = aggregateContentBlocks(taskUnitContents);
      setContentBlocks(nextBlocks);
      setContentGroups(buildContentGroups(taskUnitContents));
      setContentStatus(nextBlocks.length > 0 ? "success" : "empty");
    } catch (error) {
      if (contentRequestIdRef.current !== requestId
        || scopeRef.current.docName !== docName) {
        return;
      }
      setContentStatus("error");
      setContentError(error instanceof Error ? error.message : String(error));
    }
  }

  return {
    selectedSection,
    contentBlocks,
    contentGroups,
    contentStatus,
    contentError,
    selectSection,
    retryContent: () => selectedSection ? selectSection(selectedSection) : Promise.resolve(),
    resetContent,
    refreshSelectionLabels,
  };
}
