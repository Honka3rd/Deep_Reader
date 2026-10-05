import { useRef, useState } from "react";
import type { ContentBlock, RequestStatus, SectionSelection } from "../../types/api";
import type { TaskUnitContentService } from "../../services";
import { taskUnitContentService } from "../../services";
import {
  aggregateContentBlocks,
  buildContentGroups,
  type ReaderContentGroup,
} from "./model";

interface UseReaderContentControllerOptions {
  docName: string;
  service?: TaskUnitContentService;
}

export function useReaderContentController({
  docName,
  service = taskUnitContentService,
}: UseReaderContentControllerOptions) {
  const [selectedSection, setSelectedSection] = useState<SectionSelection | null>(null);
  const [contentBlocks, setContentBlocks] = useState<ContentBlock[]>([]);
  const [contentGroups, setContentGroups] = useState<ReaderContentGroup[]>([]);
  const [contentStatus, setContentStatus] = useState<RequestStatus>("initial");
  const [contentError, setContentError] = useState("");
  const contentRequestIdRef = useRef(0);

  function resetContent() {
    contentRequestIdRef.current += 1;
    setSelectedSection(null);
    setContentBlocks([]);
    setContentGroups([]);
    setContentError("");
    setContentStatus("initial");
  }

  async function selectSection(selection: SectionSelection) {
    const requestId = contentRequestIdRef.current + 1;
    contentRequestIdRef.current = requestId;
    setSelectedSection(selection);
    setContentBlocks([]);
    setContentGroups([]);
    setContentError("");
    if (selection.taskUnits.length === 0) {
      setContentStatus("empty");
      return;
    }

    try {
      setContentStatus("loading");
      const taskUnitContents = await service.fetchTaskUnitContents(
        docName.trim(),
        selection.taskUnits.map((taskUnit) => taskUnit.unit_id),
      );
      const nextBlocks = aggregateContentBlocks(taskUnitContents);
      if (contentRequestIdRef.current !== requestId) {
        return;
      }
      setContentBlocks(nextBlocks);
      setContentGroups(buildContentGroups(taskUnitContents));
      setContentStatus(nextBlocks.length > 0 ? "success" : "empty");
    } catch (error) {
      if (contentRequestIdRef.current !== requestId) {
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
    resetContent,
  };
}
