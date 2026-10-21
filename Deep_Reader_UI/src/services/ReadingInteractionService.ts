import type {
  AnalysisInteractionResponse,
  CriticalThinkingSessionResponse,
  QuizInteractionResponse,
  ReadingInteractionTargetRequest,
} from "../types/api";
import { RestClient } from "./RestClient";
import { assertInteractionIdentity } from "../shared/readerIdentity";

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

  private async requestInteraction<T>(path: string, options: RequestInit): Promise<T> {
    const { ok, status, payload } = await this.restClient.requestJsonWithStatus<T>(path, options);
    if (ok) return payload;
    // Recoverable interaction envelopes carry state even when HTTP is non-2xx.
    const body = payload as { envelope?: { status?: string }; detail?: unknown } | null;
    const recoverable = ["stale_target", "insufficient_content", "generation_failed", "evaluation_failed", "validation_failed"];
    if ([409, 422, 502, 503].includes(status) && recoverable.includes(body?.envelope?.status || "")) {
      return payload;
    }
    throw new Error(typeof body?.detail === "string" ? body.detail : `HTTP ${status}`);
  }

  readInsight(target: ReadingInteractionTargetRequest): Promise<AnalysisInteractionResponse> {
    assertInteractionIdentity(target);
    return this.requestInteraction<AnalysisInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<AnalysisInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<AnalysisInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<QuizInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<QuizInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<QuizInteractionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<CriticalThinkingSessionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<CriticalThinkingSessionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<CriticalThinkingSessionResponse>(
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
    assertInteractionIdentity(target);
    return this.requestInteraction<CriticalThinkingSessionResponse>(
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
