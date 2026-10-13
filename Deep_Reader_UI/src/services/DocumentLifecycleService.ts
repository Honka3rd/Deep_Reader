import type {
  DocumentLifecycleActionResponse,
} from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentLifecycleService {
  constructor(private readonly restClient: RestClient) {}

  async deleteDocument(docName: string): Promise<DocumentLifecycleActionResponse> {
    return this.restClient.requestJson<DocumentLifecycleActionResponse>(
      "/documents/lifecycle/delete",
      {
        method: "POST",
        body: JSON.stringify({
          doc_name: docName,
          action: "delete",
          confirm: true,
        }),
      },
    );
  }

  async renameDocument(
    docName: string,
    newDocName: string,
  ): Promise<DocumentLifecycleActionResponse> {
    return this.restClient.requestJson<DocumentLifecycleActionResponse>(
      "/documents/lifecycle/rename",
      {
        method: "POST",
        body: JSON.stringify({
          doc_name: docName,
          action: "rename",
          new_doc_name: newDocName,
        }),
      },
    );
  }
}
