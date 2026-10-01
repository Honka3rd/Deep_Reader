import type {
  ManualStructureValidationRequest,
  ManualStructureValidationResponse,
  ReparseDocumentStructureResponse,
} from "../types/api";
import { RestClient } from "./RestClient";

export class ManualStructureService {
  constructor(private readonly restClient: RestClient) {}

  validateManualStructure(
    request: ManualStructureValidationRequest,
  ): Promise<ManualStructureValidationResponse> {
    return this.restClient.requestJson<ManualStructureValidationResponse>(
      "/documents/manual-structure/validate",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
    );
  }

  commitManualStructure(
    request: ManualStructureValidationRequest,
  ): Promise<ReparseDocumentStructureResponse> {
    return this.restClient.requestJson<ReparseDocumentStructureResponse>(
      "/documents/reparse-structure",
      {
        method: "POST",
        body: JSON.stringify({
          doc_name: request.doc_name,
          parser_mode: "manual_structure",
          manual_structure: request.manual_structure,
        }),
      },
    );
  }
}
