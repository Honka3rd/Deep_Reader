import type { DocumentTaskLayout, PdfExtractionMode, PreparedTaskLayout } from "../types/api";
import { RestClient } from "./RestClient";
import type { TocHint } from "../types/api";

interface TaskLayoutRequestOptions {
  includeAnchorPageEvidence?: boolean;
  signal?: AbortSignal;
}

interface PrepareTaskLayoutOptions extends TaskLayoutRequestOptions {
  tocHint?: TocHint;
  layoutOnly?: boolean;
  pdfExtractionMode?: PdfExtractionMode;
}

export class TaskLayoutService {
  constructor(private readonly restClient: RestClient) {}

  fetchTaskLayout(
    docName: string,
    options: TaskLayoutRequestOptions = {},
  ): Promise<DocumentTaskLayout> {
    return this.restClient.requestJson<DocumentTaskLayout>("/documents/task-layout", {
      method: "POST",
      signal: options.signal,
      body: JSON.stringify({
        doc_name: docName,
        refresh_task_units: false,
        task_unit_split_mode: "progressive",
        include_anchor_page_evidence: Boolean(options.includeAnchorPageEvidence),
      }),
    });
  }

  prepareTaskLayout(
    docName: string,
    options: PrepareTaskLayoutOptions = {},
  ): Promise<PreparedTaskLayout> {
    return this.restClient.requestJson<PreparedTaskLayout>("/documents/prepare-task-layout", {
      method: "POST",
      signal: options.signal,
      body: JSON.stringify({
        doc_name: docName,
        force_rebuild: false,
        layout_only: Boolean(options.layoutOnly),
        ...(!options.layoutOnly && options.pdfExtractionMode ? { pdf_extraction_mode: options.pdfExtractionMode } : {}),
        structured_parser_mode: "common",
        ...(!options.layoutOnly ? { toc_hint: options.tocHint || "auto" } : {}),
        refresh_task_units: false,
        task_unit_split_mode: "progressive",
        include_anchor_page_evidence: Boolean(options.includeAnchorPageEvidence),
      }),
    });
  }
}
