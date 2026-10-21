import type {
  DocumentLifecycleActionResponse,
  DocumentManagementItem,
  DocumentUploadResponse,
  RequestStatus,
  PdfExtractionMode,
  TocHint,
} from "../../types/api";

export type FileUploadStatus =
  | "idle"
  | "validating"
  | "selected"
  | "uploading"
  | "uploaded"
  | "conflict"
  | "failed"
  | "cancelled";

export interface FileUploadDraft {
  tocHint?: TocHint;
  pdfMode?: PdfExtractionMode;
  file: File;
  fileName: string;
  fileSizeBytes: number;
  fileType: string;
}

export interface FileUploadResult {
  tocHint?: TocHint;
  pdfMode?: PdfExtractionMode;
  docName: string;
  sourceFilename: string;
  message: string;
  prepareStatus: RequestStatus | "queued" | "cancelled";
  prepareMessage: string;
  needsReconciliation?: boolean;
}

export interface DocumentPrepareState {
  tocHint?: TocHint;
  item: DocumentManagementItem;
  pdfMode?: PdfExtractionMode;
  status: RequestStatus;
  message: string;
  readerReady: boolean;
}

export interface FileUploadConflictState {
  open: boolean;
  fileName: string;
  message: string;
}

export type DocumentDeleteStatus = "idle" | "confirming" | "deleting" | "success" | "error";

export interface DocumentDeleteState {
  status: DocumentDeleteStatus;
  item: DocumentManagementItem | null;
  message: string;
}

export type DocumentRenameStatus = "idle" | "editing" | "renaming" | "success" | "error";

export interface DocumentRenameState {
  status: DocumentRenameStatus;
  item: DocumentManagementItem | null;
  draftName: string;
  message: string;
}

export interface DocumentManagementFilters {
  query: string;
  language: string;
  preparationStatus: string;
  profileLabel: string;
  sourceStatus: string;
  sourceType: string;
}

export interface DocumentManagementFilterOptions {
  languages: string[];
  preparationStatuses: string[];
  profileLabels: string[];
  sourceStatuses: string[];
  sourceTypes: string[];
}

export interface RecentDocumentShortcut {
  docName: string;
  title: string;
  openedAt: string;
}

export const defaultDocumentManagementFilters: DocumentManagementFilters = {
  query: "",
  language: "",
  preparationStatus: "",
  profileLabel: "",
  sourceStatus: "",
  sourceType: "",
};

export function getDocumentManagementTitle(item: DocumentManagementItem): string {
  return item.title?.trim() || item.docName;
}

export function getDocumentManagementSourceLabel(item: DocumentManagementItem): string {
  return item.source?.trim() || "source";
}

export function getDocumentManagementDetailLabel(item: DocumentManagementItem): string {
  const profileSummary = item.profileSummary?.trim();
  if (profileSummary) {
    return profileSummary;
  }

  const profileParts = [
    item.author?.trim(),
    item.language?.trim(),
    item.profileLabel?.trim(),
  ].filter(Boolean);
  if (profileParts.length > 0) {
    return profileParts.join(" · ");
  }

  if (item.preparationStatus) {
    return getDocumentPreparationStatusLabel(item.preparationStatus);
  }

  return "Profile details unavailable";
}

export function getDocumentPreparationStatusLabel(
  status: DocumentManagementItem["preparationStatus"],
): string {
  switch (status) {
    case "ready":
      return "Structure prepared";
    case "not_prepared":
      return "Not prepared";
    case "failed":
      return "Preparation failed";
    case "stale":
      return "Needs refresh";
    case "unknown":
    default:
      return "Readiness unknown";
  }
}

export function getDocumentSourceStatusLabel(
  status: DocumentManagementItem["sourceStatus"],
): string {
  switch (status) {
    case "available":
      return "Source available";
    case "missing":
      return "Source missing";
    case "unsupported":
      return "Source unsupported";
    case "failed":
      return "Source failed";
    default:
      return "Source status unknown";
  }
}

export function getDocumentSourceTypeLabel(
  sourceType: DocumentManagementItem["sourceType"],
): string {
  switch (sourceType) {
    case "pdf":
      return "PDF";
    case "txt":
      return "TXT";
    case "unknown":
    default:
      return "Unknown type";
  }
}

export function getDocumentManagementFilterOptions(
  items: DocumentManagementItem[],
): DocumentManagementFilterOptions {
  return {
    languages: getUniqueSortedValues(items, (item) => item.language),
    preparationStatuses: getUniqueSortedValues(items, (item) => item.preparationStatus),
    profileLabels: getUniqueSortedValues(items, (item) => item.profileLabel),
    sourceStatuses: getUniqueSortedValues(items, (item) => item.sourceStatus),
    sourceTypes: getUniqueSortedValues(items, (item) => item.sourceType),
  };
}

export function getActiveDocumentManagementFilterCount(
  filters: DocumentManagementFilters,
): number {
  return Object.values(filters).filter((value) => value.trim().length > 0).length;
}

export function filterDocumentManagementItems(
  items: DocumentManagementItem[],
  filters: DocumentManagementFilters,
): DocumentManagementItem[] {
  const query = filters.query.trim().toLowerCase();

  return items.filter((item) => {
    if (
      query &&
      ![
        item.docName,
        item.title,
        item.author,
        item.language,
        item.profileLabel,
        item.profileSummary,
        item.sourceFilename,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    ) {
      return false;
    }

    if (filters.language && item.language !== filters.language) {
      return false;
    }
    if (filters.preparationStatus && item.preparationStatus !== filters.preparationStatus) {
      return false;
    }
    if (filters.profileLabel && item.profileLabel !== filters.profileLabel) {
      return false;
    }
    if (filters.sourceStatus && item.sourceStatus !== filters.sourceStatus) {
      return false;
    }
    if (filters.sourceType && item.sourceType !== filters.sourceType) {
      return false;
    }

    return true;
  });
}

function getUniqueSortedValues(
  items: DocumentManagementItem[],
  selectValue: (item: DocumentManagementItem) => string | null | undefined,
): string[] {
  return Array.from(
    new Set(
      items
        .map((item) => selectValue(item)?.trim())
        .filter((value): value is string => Boolean(value)),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

export function upsertRecentDocumentShortcut(
  currentItems: RecentDocumentShortcut[],
  docName: string,
  catalogItem?: DocumentManagementItem,
): RecentDocumentShortcut[] {
  const trimmedDocName = docName.trim();
  if (!trimmedDocName) {
    return currentItems;
  }

  const nextItem: RecentDocumentShortcut = {
    docName: trimmedDocName,
    title: catalogItem ? getDocumentManagementTitle(catalogItem) : trimmedDocName,
    openedAt: new Date().toISOString(),
  };

  return [
    nextItem,
    ...currentItems.filter((item) => item.docName !== trimmedDocName),
  ].slice(0, 3);
}

export function formatUploadFileSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function getUploadStatusLabel(status: FileUploadStatus): string {
  switch (status) {
    case "validating":
      return "Validating";
    case "selected":
      return "Selected";
    case "uploading":
      return "Uploading";
    case "uploaded":
      return "Uploaded";
    case "conflict":
      return "Needs confirmation";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "idle":
    default:
      return "Idle";
  }
}

export function getDocumentUploadResultMessage(response: DocumentUploadResponse): string {
  if (response.source_status === "uploaded") {
    if (response.conflict_policy === "create_distinct") {
      return `Uploaded as ${response.source_filename}.`;
    }
    return `Uploaded ${response.source_filename}.`;
  }

  switch (response.reason) {
    case "missing_source":
      return "Upload failed: no source file was provided.";
    case "unsupported_file_type":
      return "Upload failed: only TXT and PDF files are supported.";
    case "file_too_large":
      return "Upload failed: the file is too large.";
    case "duplicate_conflict":
      return "Upload failed: a document with that source name already exists.";
    case "unsafe_filename":
      return "Upload failed: the filename is unsafe.";
    case "storage_failure":
      return "Upload failed: backend storage could not save the file.";
    case "validation_failure":
      return "Upload failed: backend validation rejected the file.";
    default:
      return response.reason
        ? `Upload failed: ${response.reason}.`
        : "Upload failed.";
  }
}

export function getDocumentLifecycleResultMessage(
  response: DocumentLifecycleActionResponse,
): string {
  if (response.status === "accepted" && response.deleted) {
    const affected: string[] = [];
    if (response.source_affected) {
      affected.push("source");
    }
    if (response.structured_artifacts_affected) {
      affected.push("structured artifacts");
    }
    if (response.derived_resources_affected) {
      affected.push("derived resources");
    }
    return affected.length > 0
      ? `Deleted ${response.doc_name}: ${affected.join(", ")}.`
      : `Deleted ${response.doc_name}.`;
  }

  if (response.reason) {
    return `Delete was not accepted: ${response.reason}.`;
  }

  return `Delete finished with status: ${response.status}.`;
}

export function getDocumentRenameResultMessage(
  response: DocumentLifecycleActionResponse,
): string {
  if (response.status === "accepted" && response.new_doc_name) {
    const affected: string[] = [];
    if (response.source_affected) {
      affected.push("source");
    }
    if (response.structured_artifacts_affected) {
      affected.push("structured artifacts");
    }
    if (response.derived_resources_affected) {
      affected.push("derived resources");
    }
    const suffix = affected.length > 0 ? `: ${affected.join(", ")}.` : ".";
    return `Renamed ${response.doc_name} to ${response.new_doc_name}${suffix}`;
  }

  if (response.reason) {
    return `Rename was not accepted: ${response.reason}.`;
  }

  return `Rename finished with status: ${response.status}.`;
}
