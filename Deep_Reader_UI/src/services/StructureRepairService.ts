import type {
  ReparseDocumentStructureResponse,
  StructureParserMode,
  TocHint,
  EnhancementIntent,
  EnhancementPreflightResponse,
  EnhancementCapacity,
  EnhancementOperationResponse,
} from "../types/api";
import { RestClient } from "./RestClient";

export class StructureRepairService {
  constructor(private readonly restClient: RestClient) {}

  enhancementResult(docName: string, tocHint: TocHint, confirmationToken: string, signal?: AbortSignal) {
    return this.restClient.requestJson<EnhancementOperationResponse>(
      "/documents/reparse-structure/result", {
        method: "POST", signal,
        body: JSON.stringify({ doc_name: docName, toc_hint: tocHint,
          enhancement_intent: "restore_source", confirmation_token: confirmationToken }),
      });
  }

  preflightEnhancement(docName: string, tocHint: TocHint, signal?: AbortSignal) {
    return this.restClient.requestJson<EnhancementPreflightResponse>(
      "/documents/reparse-structure/preflight", {
        method: "POST", signal,
        body: JSON.stringify({ doc_name: docName, parser_mode: "llm_enhanced",
          toc_hint: tocHint, enhancement_intent: "restore_source" }),
      });
  }

  async enhance(docName: string, tocHint: TocHint, confirmationToken: string, signal: AbortSignal) {
    const response = await this.restClient.requestJsonWithStatus<Partial<ReparseDocumentStructureResponse> & {
      detail?: string; capacity?: EnhancementCapacity | null;
    }>(
      "/documents/reparse-structure", {
        method: "POST", signal,
        body: JSON.stringify({ doc_name: docName, parser_mode: "llm_enhanced",
          toc_hint: tocHint, enhancement_intent: "restore_source",
          discard_interactions: true, confirmation_token: confirmationToken }),
      });
    return response;
  }

  reparseDocumentStructure(
    docName: string,
    parserMode: StructureParserMode,
    options: { tocHint?: TocHint; enhancementIntent?: EnhancementIntent } = {},
  ): Promise<ReparseDocumentStructureResponse> {
    if (parserMode === "llm_enhanced") {
      return Promise.reject(new Error("Enhancement requires preflight and explicit discard consent."));
    }
    return this.restClient.requestJson<ReparseDocumentStructureResponse>(
      "/documents/reparse-structure",
      {
        method: "POST",
        body: JSON.stringify({
          doc_name: docName,
          parser_mode: parserMode,
          toc_hint: options.tocHint || "auto",
          enhancement_intent: options.enhancementIntent || "restore_source",
        }),
      },
    );
  }
}
