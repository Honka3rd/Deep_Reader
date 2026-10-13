import type { DocumentUploadConflictPolicy, DocumentUploadResponse } from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentUploadService {
  constructor(private readonly restClient: RestClient) {}

  async uploadDocument(
    file: File,
    conflictPolicy: DocumentUploadConflictPolicy = "reject",
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file, file.name);
    formData.append("conflict_policy", conflictPolicy);

    const response = await this.restClient.requestJsonWithStatus<DocumentUploadResponse>(
      "/documents/upload",
      {
        method: "POST",
        body: formData,
      },
    );

    return response.payload;
  }
}
