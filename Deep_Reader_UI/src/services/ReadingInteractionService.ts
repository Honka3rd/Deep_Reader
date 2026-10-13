import type {
  AnalysisInteractionResponse,
  CriticalThinkingSessionResponse,
  QuizInteractionResponse,
  ReadingInteractionTargetRequest,
} from "../types/api";
import { RestClient } from "./RestClient";

interface AnalysisGenerationOptions {
  promptInstructionVersion?: string;
}

interface QuizGenerationOptions extends AnalysisGenerationOptions {
  maxItems?: number;
}

interface CriticalThinkingReadOptions {
  sessionId?: string;
}

type CriticalThinkingGenerationOptions = AnalysisGenerationOptions;

interface CriticalThinkingSubmitOptions {
  sessionId: string;
  answer: string;
}

interface CriticalThinkingRetryOptions {
  sessionId: string;
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

  readQuiz(target: ReadingInteractionTargetRequest): Promise<QuizInteractionResponse> {
    return this.restClient.requestJson<QuizInteractionResponse>(
      "/documents/reading-interactions/quiz/read",
      {
        method: "POST",
        body: JSON.stringify({ target }),
      },
    );
  }

  generateQuiz(
    target: ReadingInteractionTargetRequest,
    options: QuizGenerationOptions = {},
  ): Promise<QuizInteractionResponse> {
    return this.restClient.requestJson<QuizInteractionResponse>(
      "/documents/reading-interactions/quiz/generate",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          max_items: options.maxItems,
          prompt_instruction_version: options.promptInstructionVersion,
        }),
      },
    );
  }

  refreshQuiz(
    target: ReadingInteractionTargetRequest,
    options: QuizGenerationOptions = {},
  ): Promise<QuizInteractionResponse> {
    return this.restClient.requestJson<QuizInteractionResponse>(
      "/documents/reading-interactions/quiz/refresh",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          max_items: options.maxItems,
          prompt_instruction_version: options.promptInstructionVersion,
        }),
      },
    );
  }

  readCriticalThinkingSession(
    target: ReadingInteractionTargetRequest,
    options: CriticalThinkingReadOptions = {},
  ): Promise<CriticalThinkingSessionResponse> {
    return this.restClient.requestJson<CriticalThinkingSessionResponse>(
      "/documents/reading-interactions/critical-thinking/read",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          session_id: options.sessionId,
        }),
      },
    );
  }

  generateCriticalThinkingQuestion(
    target: ReadingInteractionTargetRequest,
    options: CriticalThinkingGenerationOptions = {},
  ): Promise<CriticalThinkingSessionResponse> {
    return this.restClient.requestJson<CriticalThinkingSessionResponse>(
      "/documents/reading-interactions/critical-thinking/generate-question",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          prompt_instruction_version: options.promptInstructionVersion,
        }),
      },
    );
  }

  submitCriticalThinkingAnswer(
    target: ReadingInteractionTargetRequest,
    options: CriticalThinkingSubmitOptions,
  ): Promise<CriticalThinkingSessionResponse> {
    return this.restClient.requestJson<CriticalThinkingSessionResponse>(
      "/documents/reading-interactions/critical-thinking/submit-answer",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          session_id: options.sessionId,
          answer: options.answer,
        }),
      },
    );
  }

  retryCriticalThinkingEvaluation(
    target: ReadingInteractionTargetRequest,
    options: CriticalThinkingRetryOptions,
  ): Promise<CriticalThinkingSessionResponse> {
    return this.restClient.requestJson<CriticalThinkingSessionResponse>(
      "/documents/reading-interactions/critical-thinking/retry-evaluation",
      {
        method: "POST",
        body: JSON.stringify({
          target,
          session_id: options.sessionId,
        }),
      },
    );
  }
}
