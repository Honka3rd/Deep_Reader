import { useMemo, useRef, useState } from "react";
import type { DocumentListItem } from "../../types/api";
import type { DocumentCatalogService } from "../../services";
import { documentCatalogService } from "../../services";
import { toDocumentOptions } from "./model";

interface UseBookSearchControllerOptions {
  service?: DocumentCatalogService;
}

export function useBookSearchController({
  service = documentCatalogService,
}: UseBookSearchControllerOptions = {}) {
  const [searchInput, setSearchInput] = useState("");
  const [backendDocuments, setBackendDocuments] = useState<DocumentListItem[]>([]);
  const [searching, setSearching] = useState(false);
  const requestIdRef = useRef(0);

  const documentOptions = useMemo(
    () => toDocumentOptions(backendDocuments),
    [backendDocuments],
  );

  async function loadDocumentOptions() {
    if (searching) {
      return;
    }
    setSearching(true);
    const requestId = ++requestIdRef.current;
    try {
      const response = await service.listDocuments("", 200);
      if (requestId === requestIdRef.current) setBackendDocuments(response.items);
    } catch {
      if (requestId === requestIdRef.current) setBackendDocuments([]);
    } finally {
      if (requestId === requestIdRef.current) setSearching(false);
    }
  }

  function removeDocumentOption(deletedDocName: string) {
    requestIdRef.current += 1;
    setSearching(false);
    setBackendDocuments((items) => items.filter((item) => item.doc_name !== deletedDocName));
    setSearchInput((current) => current === deletedDocName ? "" : current);
  }

  return {
    searchInput,
    setSearchInput,
    documentOptions,
    searching,
    loadDocumentOptions,
    removeDocumentOption,
  };
}
