import type { PdfExtractionMode, PreparedTaskLayout, PrepareDocumentResponse, StructureParserMode } from "../types/api";
import { RestClient } from "./RestClient";
import { TaskLayoutService } from "./TaskLayoutService";
import type { TocHint } from "../types/api";

export class DocumentPreparationService {
  constructor(private readonly restClient: RestClient) {}

  async prepareReader(docName: string, pdfExtractionMode?: PdfExtractionMode, signal?: AbortSignal, tocHint: TocHint = "auto"): Promise<PreparedTaskLayout> {
    const result = await new TaskLayoutService(this.restClient).prepareTaskLayout(docName, { pdfExtractionMode, signal, tocHint });
    if (!result.document_id || !Array.isArray(result.chapters)) {
      throw new Error("Preparation returned an invalid layout.");
    }
    if (pdfExtractionMode && (result.pdf_extraction?.requested_mode !== pdfExtractionMode || result.pdf_extraction?.effective_mode !== pdfExtractionMode ||
      !["applied", "reused"].includes(result.pdf_extraction.provenance))) {
      throw new Error("Preparation did not confirm the selected PDF mode.");
    }
    return result;
  }

  async isReaderReady(docName: string, signal?: AbortSignal): Promise<boolean> {
    try {
      const layout = await new TaskLayoutService(this.restClient).fetchTaskLayout(docName, { signal });
      return Boolean(layout.document_id && Array.isArray(layout.chapters));
    } catch {
      return false;
    }
  }

  prepareDocument(
    docName: string,
    parserMode: StructureParserMode = "common",
  ): Promise<PrepareDocumentResponse> {
    return this.restClient.requestJson<PrepareDocumentResponse>("/documents/prepare", {
      method: "POST",
      body: JSON.stringify({
        doc_name: docName,
        mode: "base",
        force_rebuild: false,
        structured_parser_mode: parserMode,
      }),
    });
  }
}
