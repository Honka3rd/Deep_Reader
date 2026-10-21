import { useEffect, useMemo, useRef, useState } from "react";
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
  PdfExtractionMode,
  TocHint,
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
  type DocumentPrepareState,
} from "./model";

interface UseFileManagementControllerOptions {
  onDocumentPreparationSettled?: (docName: string) => void;
  onDocumentDeleted?: (docName: string) => void;
  lifecycleService?: DocumentLifecycleService;
  preparationService?: DocumentPreparationService;
  service?: DocumentManagementService;
  uploadService?: DocumentUploadService;
}

export function useFileManagementController({
  onDocumentPreparationSettled,
  onDocumentDeleted,
  lifecycleService = documentLifecycleService,
  preparationService = documentPreparationService,
  service = documentManagementService,
  uploadService = documentUploadService,
}: UseFileManagementControllerOptions = {}) {
  const [open, setOpen] = useState(false);
  const listRequestIdRef = useRef(0);
  const uploadRequestRef = useRef(0);
  const uploadPdfModeRef = useRef<PdfExtractionMode>("fast");
  const uploadBusyRef = useRef(false);
  const prepareBusyRef = useRef(false);
  const openingRequest = useRef<AbortController | null>(null);
  const blockedDocuments = useRef(new Set<string>());
  const [openingDocName, setOpeningDocName] = useState<string | null>(null);
  const [openError, setOpenError] = useState("");
  const uncertainPreparations = useRef(new Set<string>());
  const uploadPreparation = useRef<{
    snapshot: FileUploadResult;
    abort: AbortController;
    timer?: ReturnType<typeof setTimeout>;
    dispatched: boolean;
  } | null>(null);
  useEffect(() => () => {
    const operation = uploadPreparation.current;
    uploadPreparation.current = null;
    if (operation?.timer) clearTimeout(operation.timer);
    operation?.abort.abort();
    openingRequest.current?.abort();
    openingRequest.current = null;
    ++uploadRequestRef.current;
  }, []);
  const [prepareState, setPrepareState] = useState<DocumentPrepareState | null>(null);
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
    invalidateOpen();
    const requestId = ++listRequestIdRef.current;
    setStatus("loading");
    setError("");
    try {
      const response = await service.listDocuments("", 200);
      if (requestId !== listRequestIdRef.current) return;
      setItems(response.items);
      setStatus(response.items.length > 0 ? "success" : "empty");
    } catch (caughtError) {
      if (requestId !== listRequestIdRef.current) return;
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
    invalidateOpen();
    if (!prepareBusyRef.current) setPrepareState(null);
    setOpen(false);
  }

  function invalidateOpen() {
    openingRequest.current?.abort();
    openingRequest.current = null;
    setOpeningDocName(null);
    setOpenError("");
  }

  function canOpenDocument(docName: string) {
    if (!docName.trim() || prepareBusyRef.current || uploadBusyRef.current ||
      deleteState.item !== null || renameState.item !== null ||
      blockedDocuments.current.has(docName) || uncertainPreparations.current.has(docName)) return false;
    if (uploadResult?.docName === docName) return uploadResult.prepareStatus === "success";
    if (prepareState?.item.docName === docName && prepareState.readerReady) return true;
    return status === "success" && items.some(item => item.docName === docName && item.preparationStatus === "ready");
  }

  async function validateDocumentOpen(docName: string): Promise<boolean> {
    if (openingRequest.current || !canOpenDocument(docName)) return false;
    const request = new AbortController();
    openingRequest.current = request;
    setOpeningDocName(docName);
    setOpenError("");
    const ready = await preparationService.isReaderReady(docName, request.signal);
    if (openingRequest.current !== request) return false;
    openingRequest.current = null;
    setOpeningDocName(null);
    if (!ready) {
      const item = items.find(candidate => candidate.docName === docName);
      if (item && item.sourceStatus === "available") {
        requestPrepareDocument(item);
      }
      setOpenError("Reader layout is unavailable. Prepare the document explicitly or retry after checking the connection.");
      return false;
    }
    return true;
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
    if (uploadBusyRef.current || prepareBusyRef.current) return;
    invalidateOpen();
    ++uploadRequestRef.current;
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
      pdfMode: extension === "pdf" ? uploadPdfModeRef.current : undefined,
    });
    setUploadStatus("selected");
  }

  function cancelUploadDraft() {
    if (uploadBusyRef.current || prepareBusyRef.current) return;
    invalidateOpen();
    ++uploadRequestRef.current;
    setUploadDraft(null);
    setUploadError("");
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadResult(null);
    setUploadStatus("cancelled");
  }

  async function uploadSelectedFile(
    conflictPolicy: DocumentUploadConflictPolicy = "reject",
  ) {
    if (!uploadDraft || uploadBusyRef.current || prepareBusyRef.current) {
      return;
    }
    uploadBusyRef.current = true;
    invalidateOpen();
    const requestId = ++uploadRequestRef.current;
    const draft = uploadDraft;

    setUploadError("");
    setUploadConflict({ open: false, fileName: "", message: "" });
    setUploadStatus("uploading");

    try {
      const response = await uploadService.uploadDocument(draft.file, conflictPolicy);
      if (requestId !== uploadRequestRef.current) return;
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
      const result: FileUploadResult = {
        docName: response.doc_name,
        sourceFilename: response.source_filename,
        message,
        prepareStatus: "queued",
        prepareMessage: "Preparation queued.",
        pdfMode: draft.pdfMode,
        tocHint: draft.tocHint || "auto",
      };
      if (!result.docName?.trim()) throw new Error("Upload returned no document name.");
      setUploadResult(result);
      setUploadStatus("uploaded");
      queueUploadedPreparation(result);
      void loadDocuments();
    } catch (caughtError) {
      if (requestId !== uploadRequestRef.current) return;
      setUploadError(caughtError instanceof Error ? caughtError.message : String(caughtError));
      setUploadStatus("failed");
    } finally {
      uploadBusyRef.current = false;
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

  function queueUploadedPreparation(snapshot: FileUploadResult) {
    invalidateOpen();
    blockedDocuments.current.add(snapshot.docName);
    prepareBusyRef.current = true;
    const operation = { snapshot, abort: new AbortController(), dispatched: false,
      timer: undefined as ReturnType<typeof setTimeout> | undefined };
    uploadPreparation.current = operation;
    setUploadResult({ ...snapshot, prepareStatus: "queued", prepareMessage: "Preparation queued." });
    operation.timer = setTimeout(() => { void runUploadedPreparation(operation); }, 0);
  }

  async function runUploadedPreparation(operation: NonNullable<typeof uploadPreparation.current>) {
    if (uploadPreparation.current !== operation) return;
    const { snapshot, abort } = operation;
    const reconcile = uncertainPreparations.current.has(snapshot.docName);
    setUploadResult({ ...snapshot, prepareStatus: "loading",
      prepareMessage: reconcile ? "Checking reader readiness." : "Preparing document." });
    try {
      if (reconcile) {
        const ready = await preparationService.isReaderReady(snapshot.docName, abort.signal);
        if (uploadPreparation.current !== operation) return;
        if (!ready) {
          setUploadResult({ ...snapshot, prepareStatus: "cancelled", needsReconciliation: true,
            prepareMessage: "Reader is not ready. Server work may still be running; no new preparation was submitted." });
          return;
        }
        uncertainPreparations.current.delete(snapshot.docName);
      } else {
        operation.dispatched = true;
        await preparationService.prepareReader(snapshot.docName, snapshot.pdfMode, abort.signal, snapshot.tocHint);
      }
      if (uploadPreparation.current !== operation) return;
      blockedDocuments.current.delete(snapshot.docName);
      setUploadResult({ ...snapshot, prepareStatus: "success", needsReconciliation: false,
        prepareMessage: "Document prepared. You can open it now." });
      void loadDocuments();
    } catch (caughtError) {
      if (uploadPreparation.current !== operation) return;
      // A lost connection is not evidence that server work stopped.
      const uncertain = caughtError instanceof TypeError;
      if (uncertain) uncertainPreparations.current.add(snapshot.docName);
      setUploadResult({ ...snapshot, prepareStatus: "error", needsReconciliation: uncertain,
        prepareMessage: caughtError instanceof Error ? caughtError.message : String(caughtError) });
    } finally {
      if (uploadPreparation.current === operation) {
        uploadPreparation.current = null;
        prepareBusyRef.current = false;
      }
    }
  }

  function prepareUploadedDocument() {
    if (!uploadResult || prepareBusyRef.current || uploadBusyRef.current) return;
    queueUploadedPreparation(uploadResult);
  }

  function cancelUploadedPreparation() {
    const operation = uploadPreparation.current;
    if (!operation) return;
    uploadPreparation.current = null;
    clearTimeout(operation.timer);
    operation.abort.abort();
    prepareBusyRef.current = false;
    const { snapshot, dispatched } = operation;
    if (dispatched) uncertainPreparations.current.add(snapshot.docName);
    const uncertain = uncertainPreparations.current.has(snapshot.docName);
    setUploadResult({ ...snapshot, prepareStatus: "cancelled", needsReconciliation: uncertain,
      prepareMessage: uncertain ? "Waiting cancelled. Server preparation may continue."
        : "Preparation cancelled before submission." });
  }

  function retryUpload() {
    if (uploadBusyRef.current || prepareBusyRef.current) return;
    invalidateOpen();
    if ((uploadStatus === "failed" || uploadStatus === "conflict") && uploadDraft) {
      void uploadSelectedFile();
      return;
    }
    setUploadDraft(null);
    ++uploadRequestRef.current;
    setUploadError("");
    setUploadResult(null);
    setUploadStatus("idle");
  }

  function requestDeleteDocument(item: DocumentManagementItem) {
    if (prepareBusyRef.current || uploadBusyRef.current) return;
    invalidateOpen();
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
      listRequestIdRef.current += 1;
      blockedDocuments.current.add(item.docName);
      setItems((current) => current.filter((candidate) => candidate.docName !== item.docName));
      setRecentItems((current) => current.filter((candidate) => candidate.docName !== item.docName));
      onDocumentDeleted?.(item.docName);
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
    if (prepareBusyRef.current || uploadBusyRef.current) return;
    invalidateOpen();
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
      blockedDocuments.current.add(item.docName);
      setRecentItems(current => current.filter(candidate => candidate.docName !== item.docName));
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

  function changeUploadPdfMode(mode: PdfExtractionMode) {
    if (uploadBusyRef.current || prepareBusyRef.current || uploadStatus === "conflict") return;
    if (uploadResult && uncertainPreparations.current.has(uploadResult.docName)) return;
    invalidateOpen();
    if (uploadResult?.pdfMode) {
      uploadPdfModeRef.current = mode;
      setUploadResult({ ...uploadResult, pdfMode: mode, prepareStatus: "initial", prepareMessage: "" });
    } else if (uploadDraft?.pdfMode) {
      uploadPdfModeRef.current = mode;
      setUploadDraft({ ...uploadDraft, pdfMode: mode });
    }
  }

  function requestPrepareDocument(item: DocumentManagementItem) {
    if (prepareBusyRef.current || uploadBusyRef.current || item.sourceStatus !== "available") return;
    invalidateOpen();
    setPrepareState({ item, pdfMode: item.sourceType === "pdf" ? "fast" : undefined,
      status: "initial", message: "", readerReady: false });
  }

  function changePreparePdfMode(pdfMode: PdfExtractionMode) {
    if (prepareBusyRef.current) return;
    invalidateOpen();
    setPrepareState((current) => current && ({ ...current, pdfMode, status: "initial", message: "" }));
  }

  function changeUploadTocHint(tocHint: TocHint) {
    if (uploadBusyRef.current || prepareBusyRef.current || uploadStatus === "conflict") return;
    if (uploadResult && uncertainPreparations.current.has(uploadResult.docName)) return;
    if (uploadResult) setUploadResult({ ...uploadResult, tocHint, prepareStatus: "initial", prepareMessage: "" });
    else if (uploadDraft) setUploadDraft({ ...uploadDraft, tocHint });
  }

  function changePrepareTocHint(tocHint: TocHint) {
    if (!prepareBusyRef.current) setPrepareState(current => current && ({ ...current, tocHint, status: "initial", message: "" }));
  }

  function closePrepareDocument() {
    if (!prepareBusyRef.current) {
      invalidateOpen();
      setPrepareState(null);
    }
  }

  function acceptReadableDocument(docName: string) {
    blockedDocuments.current.delete(docName);
    setUploadResult(current => current?.docName === docName ? {
      ...current, prepareStatus: "success", needsReconciliation: false,
      prepareMessage: "Document prepared. You can open it now.",
    } : current);
  }

  async function prepareExistingDocument() {
    if (!prepareState || prepareBusyRef.current || uploadBusyRef.current) return;
    const snapshot = prepareState;
    invalidateOpen();
    blockedDocuments.current.add(snapshot.item.docName);
    prepareBusyRef.current = true;
    setPrepareState({ ...snapshot, status: "loading", message: "", readerReady: false });
    try {
      if (uncertainPreparations.current.has(snapshot.item.docName)) {
        const readerReady = await preparationService.isReaderReady(snapshot.item.docName);
        if (readerReady) {
          uncertainPreparations.current.delete(snapshot.item.docName);
          acceptReadableDocument(snapshot.item.docName);
        }
        setPrepareState({ ...snapshot, status: readerReady ? "success" : "error", readerReady,
          message: readerReady ? "Existing layout is available." : "Server work may still be running. No new preparation was submitted." });
        return;
      }
      await preparationService.prepareReader(snapshot.item.docName, snapshot.pdfMode, undefined, snapshot.tocHint);
      acceptReadableDocument(snapshot.item.docName);
      setPrepareState({ ...snapshot, status: "success", message: "Document prepared.", readerReady: true });
    } catch (error) {
      const readerReady = await preparationService.isReaderReady(snapshot.item.docName);
      if (readerReady) acceptReadableDocument(snapshot.item.docName);
      setPrepareState({ ...snapshot, status: "error", readerReady,
        message: `${error instanceof Error ? error.message : String(error)}${readerReady ? " Existing layout is available." : ""}` });
    } finally {
      prepareBusyRef.current = false;
      onDocumentPreparationSettled?.(snapshot.item.docName);
      await loadDocuments();
    }
  }

  return {
    openingDocName,
    openError,
    canOpenDocument,
    validateDocumentOpen,
    cancelUploadedPreparation,
    prepareState,
    changeUploadTocHint,
    changePrepareTocHint,
    changeUploadPdfMode,
    changePreparePdfMode,
    requestPrepareDocument,
    closePrepareDocument,
    prepareExistingDocument,
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
