import { useMemo, useState } from "react";
import type { DocumentListItem } from "../../types/api";
import type { DocumentCatalogService } from "../../services";
import { documentCatalogService } from "../../services";
import { isSelectedDocumentOption, toDocumentOptions } from "./model";

interface UseBookSearchControllerOptions {
  service?: DocumentCatalogService;
}

export function useBookSearchController({
  service = documentCatalogService,
}: UseBookSearchControllerOptions = {}) {
  const [docName, setDocName] = useState("");
  const [backendDocuments, setBackendDocuments] = useState<DocumentListItem[]>([]);
  const [searching, setSearching] = useState(false);

  const documentOptions = useMemo(
    () => toDocumentOptions(backendDocuments),
    [backendDocuments],
  );

  async function loadDocumentOptions() {
    if (searching) {
      return;
    }
    setSearching(true);
    try {
      const response = await service.listDocuments("", 200);
      setBackendDocuments(response.items);
    } catch {
      setBackendDocuments([]);
    } finally {
      setSearching(false);
    }
  }

  return {
    docName,
    setDocName,
    documentOptions,
    searching,
    loadDocumentOptions,
    isSelectedDocument: isSelectedDocumentOption(docName, documentOptions),
  };
}
