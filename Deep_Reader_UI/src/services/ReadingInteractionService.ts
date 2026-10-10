import type {
  AnalysisInteractionResponse,
  ReadingInteractionTargetRequest,
} from "../types/api";
import { RestClient } from "./RestClient";

interface AnalysisGenerationOptions {
  promptInstructionVersion?: string;
}

export class ReadingInteractionService {
  constructor(private readonly restClient: RestClient) {}

  readInsight(target: ReadingInteractionTargetRequest): Promise<AnalysisInteractionResponse> {
    return this.restClient.requestJson<AnalysisInteractionResponse>(
      "/documents/reading-interactions/insight/read",
      {
        method: "POST",
        body: JSON.stringify({ target }),
      },
    );
  }

  generateInsight(
    target: ReadingInteractionTargetRequest,
    options: AnalysisGenerationOptions = {},
  ): Promise<AnalysisInteractionResponse> {
    return this.restClient.requestJson<AnalysisInteractionResponse>(
      "/documents/reading-interactions/insight/generate",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          prompt_instruction_version: options.promptInstructionVersion,
        }),
      },
    );
  }

  refreshInsight(
    target: ReadingInteractionTargetRequest,
    options: AnalysisGenerationOptions = {},
  ): Promise<AnalysisInteractionResponse> {
    return this.restClient.requestJson<AnalysisInteractionResponse>(
      "/documents/reading-interactions/insight/refresh",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          prompt_instruction_version: options.promptInstructionVersion,
        }),
      },
    );
  }
}
