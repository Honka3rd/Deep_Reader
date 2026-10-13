import type {
  DocumentListItem,
  DocumentManagementItem,
  DocumentManagementListResult,
} from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentManagementService {
  constructor(private readonly restClient: RestClient) {}

  async listDocuments(query: string, limit = 200): Promise<DocumentManagementListResult> {
    const params = new URLSearchParams();
    const normalizedQuery = query.trim();
    if (normalizedQuery) {
      params.set("q", normalizedQuery);
    }
    params.set("limit", String(limit));

    const response = await this.restClient.requestJson<{
      items: DocumentListItem[];
      query?: string | null;
      total: number;
    }>(`/documents?${params.toString()}`, { method: "GET" });

    return {
      items: response.items.map(toDocumentManagementItem),
      query: response.query,
      total: response.total,
    };
  }
}

function toDocumentManagementItem(item: DocumentListItem): DocumentManagementItem {
  return {
    docName: item.doc_name,
    title: item.title,
    source: item.source,
  };
}
