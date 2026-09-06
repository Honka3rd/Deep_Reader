import type {
  DocumentListResponse,
  DocumentTaskLayout,
  PrepareDocumentResponse,
  ReparseDocumentStructureResponse,
  StructureParserMode,
  TaskUnitContent,
} from "../types/api";

const API_BASE = "/api";

async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? payload.detail
        : payload &&
            typeof payload === "object" &&
            "errors" in payload &&
            Array.isArray(payload.errors) &&
            payload.errors.length > 0
          ? payload.errors.join("; ")
        : `HTTP ${response.status}`;
    throw new Error(String(detail));
  }

  return payload as T;
}

export function fetchTaskLayout(docName: string): Promise<DocumentTaskLayout> {
  return requestJson<DocumentTaskLayout>("/documents/task-layout", {
    method: "POST",
    body: JSON.stringify({
      doc_name: docName,
      refresh_task_units: false,
      task_unit_split_mode: "progressive",
    }),
  });
}

export function prepareTaskLayout(docName: string): Promise<DocumentTaskLayout> {
  return requestJson<DocumentTaskLayout>("/documents/prepare-task-layout", {
    method: "POST",
    body: JSON.stringify({
      doc_name: docName,
      force_rebuild: false,
      structured_parser_mode: "common",
      refresh_task_units: false,
      task_unit_split_mode: "progressive",
    }),
  });
}

export function fetchDocumentList(query: string, limit = 20): Promise<DocumentListResponse> {
  const params = new URLSearchParams();
  const normalizedQuery = query.trim();
  if (normalizedQuery) {
    params.set("q", normalizedQuery);
  }
  params.set("limit", String(limit));
  return requestJson<DocumentListResponse>(`/documents?${params.toString()}`, {
    method: "GET",
  });
}

export function prepareDocument(
  docName: string,
  parserMode: StructureParserMode = "common",
): Promise<PrepareDocumentResponse> {
  return requestJson<PrepareDocumentResponse>("/documents/prepare", {
    method: "POST",
    body: JSON.stringify({
      doc_name: docName,
      mode: "base",
      force_rebuild: false,
      structured_parser_mode: parserMode,
    }),
  });
}

export function reparseDocumentStructure(
  docName: string,
  parserMode: StructureParserMode,
): Promise<ReparseDocumentStructureResponse> {
  return requestJson<ReparseDocumentStructureResponse>("/documents/reparse-structure", {
    method: "POST",
    body: JSON.stringify({
      doc_name: docName,
      parser_mode: parserMode,
    }),
  });
}

export function fetchTaskUnitContent(
  docName: string,
  taskUnitId: string,
): Promise<TaskUnitContent> {
  const encodedDocName = encodeURIComponent(docName);
  const encodedTaskUnitId = encodeURIComponent(taskUnitId);
  return requestJson<TaskUnitContent>(
    `/documents/${encodedDocName}/task-units/${encodedTaskUnitId}/content?segmented=true`,
    { method: "GET" },
  );
}
