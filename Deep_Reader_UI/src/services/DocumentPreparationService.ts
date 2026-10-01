import type { PrepareDocumentResponse, StructureParserMode } from "../types/api";
import { RestClient } from "./RestClient";

export class DocumentPreparationService {
  constructor(private readonly restClient: RestClient) {}

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
