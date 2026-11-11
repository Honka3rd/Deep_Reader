import type {
  DocumentLifecycleActionResponse,
  DocumentLifecycleActionRequest,
  DocumentRenameContextResponse,
} from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentLifecycleService {
  private readonly renameCommands = new Map<string, DocumentLifecycleActionRequest>();
  private readonly renameRequests = new Map<string, Promise<DocumentLifecycleActionResponse>>();

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

  renameDocument(docName: string, newDocName: string): Promise<DocumentLifecycleActionResponse> {
    const key = JSON.stringify([docName, newDocName]);
    const active = this.renameRequests.get(key);
    if (active) return active;
    const request = this.submitRename(key, docName, newDocName).finally(() => {
      this.renameRequests.delete(key);
    });
    this.renameRequests.set(key, request);
    return request;
  }

  private async submitRename(key: string, docName: string, newDocName: string): Promise<DocumentLifecycleActionResponse> {
    let command = this.renameCommands.get(key);
    if (!command) {
      const context = await this.restClient.requestJson<DocumentRenameContextResponse>(
        `/documents/lifecycle/rename-context?doc_name=${encodeURIComponent(docName)}`,
      );
      command = {
        doc_name: docName, action: "rename", new_doc_name: newDocName,
        command_id: crypto.randomUUID(), expected_source_token: context.source_token,
      };
      this.renameCommands.set(key, command);
    }
    // Transport/5xx outcomes retain this exact command. Only an explicit retry
    // sends it again; fetching new context could rename a reimported book.
    const { ok, payload, status } = await this.restClient.requestJsonWithStatus<DocumentLifecycleActionResponse>(
      "/documents/lifecycle/rename", { method: "POST", body: JSON.stringify(command) },
    );
    if ((ok && payload.status === "accepted") || (status >= 400 && status < 500)) {
      this.renameCommands.delete(key);
    }
    if (!ok) throw new Error(payload.reason || `HTTP ${status}`);
    return payload;
  }
}
