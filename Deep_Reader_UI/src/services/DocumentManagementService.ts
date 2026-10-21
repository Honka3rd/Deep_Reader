import type {
  DocumentListItem,
  DocumentManagementItem,
  DocumentManagementListResult,
} from "../types/api";
import { RestClient } from "./RestClient";

interface ManagementRow extends DocumentListItem {
  review_required?: boolean;
  source_type?: DocumentManagementItem["sourceType"];
  source_filename?: string | null;
  source_status?: DocumentManagementItem["sourceStatus"];
  preparation_status?: DocumentManagementItem["preparationStatus"];
}

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
      items: ManagementRow[];
      query?: string | null;
      total: number;
    }>(`/documents/management?${params.toString()}`, { method: "GET" });

    return {
      items: response.items.map(toDocumentManagementItem),
      query: response.query,
      total: response.total,
    };
  }
}

function toDocumentManagementItem(item: ManagementRow): DocumentManagementItem {
  return {
    docName: item.doc_name,
    title: item.title,
    source: item.source,
    sourceType: item.source_type,
    sourceFilename: item.source_filename,
    sourceStatus: item.source_status,
    preparationStatus: item.preparation_status,
    reviewRequired: item.review_required === true,
  };
}
