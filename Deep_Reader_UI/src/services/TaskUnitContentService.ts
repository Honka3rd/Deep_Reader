import type { BatchTaskUnitContentResponse, TaskUnitContent } from "../types/api";
import { RestClient } from "./RestClient";
import { requireBackendIdentity } from "../shared/readerIdentity";

export class TaskUnitContentService {
  constructor(private readonly restClient: RestClient) {}

  async fetchTaskUnitContent(docName: string, taskUnitId: string): Promise<TaskUnitContent> {
    requireBackendIdentity(docName, "Document name");
    requireBackendIdentity(taskUnitId, "Task-unit ID");
    const encodedDocName = encodeURIComponent(docName);
    const encodedTaskUnitId = encodeURIComponent(taskUnitId);
    const response = await this.restClient.requestJson<TaskUnitContent>(
      `/documents/${encodedDocName}/task-units/${encodedTaskUnitId}/content?segmented=true`,
      { method: "GET" },
    );
    if (response?.task_unit_id !== taskUnitId) {
      throw new Error("Content response does not match the requested task unit.");
    }
    return response;
  }

  async fetchTaskUnitContents(
    docName: string,
    taskUnitIds: string[],
    signal?: AbortSignal,
  ): Promise<TaskUnitContent[]> {
    requireBackendIdentity(docName, "Document name");
    taskUnitIds.forEach((id) => requireBackendIdentity(id, "Task-unit ID"));
    if (new Set(taskUnitIds).size !== taskUnitIds.length) throw new Error("Duplicate task-unit IDs.");
    if (taskUnitIds.length === 0) return [];
    if (taskUnitIds.length > 20) {
      const contents: TaskUnitContent[] = [];
      for (let offset = 0; offset < taskUnitIds.length; offset += 20) {
        signal?.throwIfAborted();
        contents.push(...await this.fetchTaskUnitContents(
          docName, taskUnitIds.slice(offset, offset + 20), signal));
      }
      return contents;
    }
    const encodedDocName = encodeURIComponent(docName);
    const response = await this.restClient.requestJson<BatchTaskUnitContentResponse>(
      `/documents/${encodedDocName}/task-units/content`,
      {
        method: "POST",
        signal,
        body: JSON.stringify({
          task_unit_ids: taskUnitIds,
          segmented: true,
        }),
      },
    );
    if (!Array.isArray(response.contents)
      || response.contents.length !== taskUnitIds.length
      || response.contents.some((item, index) => item?.task_unit_id !== taskUnitIds[index])) {
      throw new Error("Content response does not match the requested task units. Retry the section.");
    }
    return response.contents;
  }
}
