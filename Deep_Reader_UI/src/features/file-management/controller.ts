import { useMemo, useState } from "react";
import type {
  DocumentLifecycleService,
  DocumentManagementService,
  DocumentPreparationService,
  DocumentUploadService,
} from "../../services";
import {
  documentPreparationService,
  documentLifecycleService,
  documentManagementService,
  documentUploadService,
} from "../../services";
import type {
  DocumentManagementItem,
  DocumentUploadConflictPolicy,
  RequestStatus,
} from "../../types/api";
import {
  defaultDocumentManagementFilters,
  filterDocumentManagementItems,
  getActiveDocumentManagementFilterCount,
  getDocumentManagementFilterOptions,
  getDocumentLifecycleResultMessage,
  getDocumentRenameResultMessage,
  getDocumentUploadResultMessage,
  upsertRecentDocumentShortcut,
  type DocumentDeleteState,
  type DocumentManagementFilters,
  type DocumentRenameState,
  type FileUploadConflictState,
  type FileUploadDraft,
  type FileUploadResult,
  type FileUploadStatus,
  type RecentDocumentShortcut,
} from "./model";

interface UseFileManagementControllerOptions {
  lifecycleService?: DocumentLifecycleService;
  preparationService?: DocumentPreparationService;
  service?: DocumentManagementService;
  uploadService?: DocumentUploadService;
}

export function useFileManagementController({
  lifecycleService = documentLifecycleService,
  preparationService = documentPreparationService,
  service = documentManagementService,
  uploadService = documentUploadService,
}: UseFileManagementControllerOptions = {}) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<DocumentManagementItem[]>([]);
  const [filters, setFilters] = useState<DocumentManagementFilters>(
    defaultDocumentManagementFilters,
  );
  const [recentItems, setRecentItems] = useState<RecentDocumentShortcut[]>([]);
  const [status, setStatus] = useState<RequestStatus>("initial");
  const [error, setError] = useState("");
  const [uploadStatus, setUploadStatus] = useState<FileUploadStatus>("idle");
  const [uploadDraft, setUploadDraft] = useState<FileUploadDraft | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [uploadConflict, setUploadConflict] = useState<FileUploadConflictState>({
    open: false,
    fileName: "",
    message: "",
  });
  const [uploadResult, setUploadResult] = useState<FileUploadResult | null>(null);
  const [deleteState, setDeleteState] = useState<DocumentDeleteState>({
    status: "idle",
    item: null,
    message: "",
  });
  const [renameState, setRenameState] = useState<DocumentRenameState>({
    status: "idle",
    item: null,
    draftName: "",
    message: "",
  });

  const total = useMemo(() => items.length, [items]);
  const filteredItems = useMemo(() => filterDocumentManagementItems(items, filters), [
    filters,
    items,
  ]);
  const filterOptions = useMemo(() => getDocumentManagementFilterOptions(items), [items]);
  const activeFilterCount = useMemo(
    () => getActiveDocumentManagementFilterCount(filters),
    [filters],
  );

  async function loadDocuments() {
    setStatus("loading");
    setError("");
    try {
      const response = await service.listDocuments("", 200);
      setItems(response.items);
      setStatus(response.items.length > 0 ? "success" : "empty");
    } catch (caughtError) {
      setItems([]);
      setStatus("error");
      setError(caughtError instanceof Error ? caughtError.message : String(caughtError));
    }
  }

  function openLibrary() {
    setOpen(true);
    void loadDocuments();
  }

  function closeLibrary() {
    setOpen(false);
  }

  function changeFilter(name: keyof DocumentManagementFilters, value: string) {
    setFilters((current) => ({
      ...current,
      [name]: value,
    }));
  }

  function resetFilters() {
    setFilters(defaultDocumentManagementFilters);
  }

  function recordRecentDocument(docName: string) {
    setRecentItems((currentItems) =>
      upsertRecentDocumentShortcut(
        currentItems,
        docName,
        items.find((item) => item.docName === docName),
      ),
    );
  }

  function selectUploadFile(file: File | null) {
    setUploadError("");
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadResult(null);
    if (!file) {
      setUploadDraft(null);
      setUploadStatus("idle");
      return;
    }

    setUploadStatus("validating");
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    if (!["txt", "pdf"].includes(extension)) {
      setUploadDraft({
        file,
        fileName: file.name,
        fileSizeBytes: file.size,
        fileType: file.type || "unknown",
      });
      setUploadError("Unsupported file type");
      setUploadStatus("failed");
      return;
    }

    setUploadDraft({
      file,
      fileName: file.name,
      fileSizeBytes: file.size,
      fileType: file.type || extension,
    });
    setUploadStatus("selected");
  }

  function cancelUploadDraft() {
    setUploadDraft(null);
    setUploadError("");
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadResult(null);
    setUploadStatus("cancelled");
  }

  async function uploadSelectedFile(
    conflictPolicy: DocumentUploadConflictPolicy = "reject",
  ) {
    if (!uploadDraft || uploadStatus === "uploading") {
      return;
    }

    setUploadError("");
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadStatus("uploading");

    try {
      const response = await uploadService.uploadDocument(uploadDraft.file, conflictPolicy);
      const message = getDocumentUploadResultMessage(response);
      if (
        response.source_status === "failed" &&
        response.reason === "duplicate_conflict" &&
        conflictPolicy === "reject"
      ) {
        setUploadError(message);
        setUploadResult(null);
        setUploadConflict({
          open: true,
          fileName: response.source_filename || uploadDraft.fileName,
          message,
        });
        setUploadStatus("conflict");
        return;
      }
      if (response.source_status !== "uploaded") {
        setUploadError(message);
        setUploadResult(null);
        setUploadStatus("failed");
        return;
      }

      setUploadError(message);
      setUploadResult({
        docName: response.doc_name,
        sourceFilename: response.source_filename,
        message,
        prepareStatus: "initial",
        prepareMessage: "",
      });
      setUploadStatus("uploaded");
      await loadDocuments();
    } catch (caughtError) {
      setUploadError(caughtError instanceof Error ? caughtError.message : String(caughtError));
      setUploadStatus("failed");
    }
  }

  function cancelDuplicateUpload() {
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadError("");
    setUploadStatus(uploadDraft ? "selected" : "idle");
  }

  function confirmDuplicateUpload() {
    void uploadSelectedFile("create_distinct");
  }

  async function prepareUploadedDocument() {
    if (!uploadResult || uploadResult.prepareStatus === "loading") {
      return;
    }

    setUploadResult({
      ...uploadResult,
      prepareStatus: "loading",
      prepareMessage: "",
    });

    try {
      const response = await preparationService.prepareDocument(uploadResult.docName);
      if (!response.success || !response.structured_document_ready) {
        setUploadResult({
          ...uploadResult,
          prepareStatus: "error",
          prepareMessage:
            response.errors.length > 0
              ? response.errors.join("; ")
              : "Document preparation did not complete.",
        });
        return;
      }

      setUploadResult({
        ...uploadResult,
        prepareStatus: "success",
        prepareMessage: "Document prepared. You can open it now.",
      });
      await loadDocuments();
    } catch (caughtError) {
      setUploadResult({
        ...uploadResult,
        prepareStatus: "error",
        prepareMessage: caughtError instanceof Error ? caughtError.message : String(caughtError),
      });
    }
  }

  function retryUpload() {
    if ((uploadStatus === "failed" || uploadStatus === "conflict") && uploadDraft) {
      void uploadSelectedFile();
      return;
    }
    setUploadDraft(null);
    setUploadError("");
    setUploadResult(null);
    setUploadStatus("idle");
  }

  function requestDeleteDocument(item: DocumentManagementItem) {
    setDeleteState({
      status: "confirming",
      item,
      message: "",
    });
  }

  function cancelDeleteDocument() {
    setDeleteState({
      status: "idle",
      item: null,
      message: "",
    });
  }

  async function confirmDeleteDocument() {
    const item = deleteState.item;
    if (!item || deleteState.status === "deleting") {
      return;
    }

    setDeleteState({
      status: "deleting",
      item,
      message: "",
    });

    try {
      const response = await lifecycleService.deleteDocument(item.docName);
      const message = getDocumentLifecycleResultMessage(response);
      if (response.status !== "accepted" || !response.deleted) {
        setDeleteState({
          status: "error",
          item,
          message,
        });
        return;
      }

      setDeleteState({
        status: "success",
        item: null,
        message,
      });
      await loadDocuments();
    } catch (caughtError) {
      setDeleteState({
        status: "error",
        item,
        message: caughtError instanceof Error ? caughtError.message : String(caughtError),
      });
    }
  }

  function requestRenameDocument(item: DocumentManagementItem) {
    setRenameState({
      status: "editing",
      item,
      draftName: item.docName,
      message: "",
    });
  }

  function changeRenameDraft(nextName: string) {
    setRenameState((current) => ({
      ...current,
      draftName: nextName,
      message: current.status === "error" ? "" : current.message,
      status: current.status === "error" ? "editing" : current.status,
    }));
  }

  function cancelRenameDocument() {
    setRenameState({
      status: "idle",
      item: null,
      draftName: "",
      message: "",
    });
  }

  async function confirmRenameDocument() {
    const item = renameState.item;
    const nextName = renameState.draftName.trim();
    if (!item || renameState.status === "renaming") {
      return;
    }
    if (!nextName) {
      setRenameState({
        ...renameState,
        status: "error",
        message: "Document name is required.",
      });
      return;
    }
    if (nextName === item.docName) {
      setRenameState({
        ...renameState,
        status: "error",
        message: "Choose a different document name.",
      });
      return;
    }

    setRenameState({
      status: "renaming",
      item,
      draftName: nextName,
      message: "",
    });

    try {
      const response = await lifecycleService.renameDocument(item.docName, nextName);
      const message = getDocumentRenameResultMessage(response);
      if (response.status !== "accepted" || !response.new_doc_name) {
        setRenameState({
          status: "error",
          item,
          draftName: nextName,
          message,
        });
        return;
      }

      setRenameState({
        status: "success",
        item: null,
        draftName: "",
        message,
      });
      await loadDocuments();
    } catch (caughtError) {
      setRenameState({
        status: "error",
        item,
        draftName: nextName,
        message: caughtError instanceof Error ? caughtError.message : String(caughtError),
      });
    }
  }

  return {
    deleteState,
    error,
    activeFilterCount,
    filterOptions,
    filters,
    items: filteredItems,
    open,
    recentItems,
    renameState,
    status,
    total,
    visibleTotal: filteredItems.length,
    uploadConflict,
    uploadDraft,
    uploadError,
    uploadResult,
    uploadStatus,
    cancelUploadDraft,
    cancelDeleteDocument,
    cancelDuplicateUpload,
    cancelRenameDocument,
    closeLibrary,
    changeFilter,
    confirmDeleteDocument,
    confirmDuplicateUpload,
    confirmRenameDocument,
    changeRenameDraft,
    loadDocuments,
    openLibrary,
    prepareUploadedDocument,
    requestDeleteDocument,
    requestRenameDocument,
    recordRecentDocument,
    resetFilters,
    retryUpload,
    selectUploadFile,
    uploadSelectedFile,
  };
}
