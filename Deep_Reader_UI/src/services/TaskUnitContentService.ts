import type { BatchTaskUnitContentResponse, TaskUnitContent } from "../types/api";
import { RestClient } from "./RestClient";

export class TaskUnitContentService {
  constructor(private readonly restClient: RestClient) {}

  fetchTaskUnitContent(docName: string, taskUnitId: string): Promise<TaskUnitContent> {
    const encodedDocName = encodeURIComponent(docName);
    const encodedTaskUnitId = encodeURIComponent(taskUnitId);
    return this.restClient.requestJson<TaskUnitContent>(
      `/documents/${encodedDocName}/task-units/${encodedTaskUnitId}/content?segmented=true`,
      { method: "GET" },
    );
  }

  async fetchTaskUnitContents(
    docName: string,
    taskUnitIds: string[],
  ): Promise<TaskUnitContent[]> {
    const encodedDocName = encodeURIComponent(docName);
    const response = await this.restClient.requestJson<BatchTaskUnitContentResponse>(
      `/documents/${encodedDocName}/task-units/content`,
      {
        method: "POST",
        body: JSON.stringify({
          task_unit_ids: taskUnitIds,
          segmented: true,
        }),
      },
    );
    return response.contents;
  }
}
