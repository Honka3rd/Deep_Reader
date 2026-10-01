import type { DocumentListResponse } from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentCatalogService {
  constructor(private readonly restClient: RestClient) {}

  listDocuments(query: string, limit = 20): Promise<DocumentListResponse> {
    const params = new URLSearchParams();
    const normalizedQuery = query.trim();
    if (normalizedQuery) {
      params.set("q", normalizedQuery);
    }
    params.set("limit", String(limit));
    return this.restClient.requestJson<DocumentListResponse>(
      `/documents?${params.toString()}`,
      { method: "GET" },
    );
  }
}
