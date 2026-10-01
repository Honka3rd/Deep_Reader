import type {
  ReparseDocumentStructureResponse,
  StructureParserMode,
} from "../types/api";
import { RestClient } from "./RestClient";

export class StructureRepairService {
  constructor(private readonly restClient: RestClient) {}

  reparseDocumentStructure(
    docName: string,
    parserMode: StructureParserMode,
  ): Promise<ReparseDocumentStructureResponse> {
    return this.restClient.requestJson<ReparseDocumentStructureResponse>(
      "/documents/reparse-structure",
      {
        method: "POST",
        body: JSON.stringify({
          doc_name: docName,
          parser_mode: parserMode,
        }),
      },
    );
  }
}
