import type { DocumentTaskLayout } from "../types/api";
import { RestClient } from "./RestClient";

interface TaskLayoutRequestOptions {
  includeAnchorPageEvidence?: boolean;
}

export class TaskLayoutService {
  constructor(private readonly restClient: RestClient) {}

  fetchTaskLayout(
    docName: string,
    options: TaskLayoutRequestOptions = {},
  ): Promise<DocumentTaskLayout> {
    return this.restClient.requestJson<DocumentTaskLayout>("/documents/task-layout", {
      method: "POST",
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
    options: TaskLayoutRequestOptions = {},
  ): Promise<DocumentTaskLayout> {
    return this.restClient.requestJson<DocumentTaskLayout>("/documents/prepare-task-layout", {
      method: "POST",
      body: JSON.stringify({
        doc_name: docName,
        force_rebuild: false,
        structured_parser_mode: "common",
        refresh_task_units: false,
        task_unit_split_mode: "progressive",
        include_anchor_page_evidence: Boolean(options.includeAnchorPageEvidence),
      }),
    });
  }
}
