import type { ReadingInteractionTargetRequest, ReadingInteractionResponseEnvelope } from "../../types/api";
import type { ReadingInteractionService } from "../../services/ReadingInteractionService";
import { readOrGenerate, validateInteractionResponse } from "./activation";
import { InteractionGenerationTasks } from "./generation-tasks";
import { useEffect, useRef, useState } from "react";
import {
  mapAnalysisInteractionResponseToInsightViewState,
  mapQuizInteractionResponseToQuizViewState,
  mapCriticalThinkingSessionResponseToViewState,
  normalizeReadingInteractionThrownError,
  buildReadingInteractionBreadcrumb,
  buildReadingInteractionTargetKey,
  createInitialCriticalThinkingViewState,
  createInitialInsightViewState,
  createInitialQuizViewState,
  type CriticalThinkingStateByTarget,
  type CriticalThinkingStatus,
  type CriticalThinkingViewState,
  type InlineInsightStateByTarget,
  type InsightViewState,
  type InteractionStatus,
  type QuizViewState,
  type QuizStateByTarget,
  type ReadingInteractionKind,
  type ReadingInteractionTarget,
  type ReadingInteractionTargetKey,
} from "./model";

export type ReadingInteractionSurface = "inline_insight" | "drawer";

export interface ReadingInteractionMenuSelection {
  kind: ReadingInteractionKind;
  target: ReadingInteractionTarget;
  targetKey: ReadingInteractionTargetKey;
  breadcrumb: string[];
  surface: ReadingInteractionSurface;
  requestId: number;
}

function surfaceForInteractionKind(
  kind: ReadingInteractionKind,
): ReadingInteractionSurface {
  return kind === "insight" ? "inline_insight" : "drawer";
}

export function useReadingInteractionMenuController() {
  const [interactionMenuTarget, setInteractionMenuTarget] =
    useState<ReadingInteractionTarget | null>(null);
  const [selectedInteraction, setSelectedInteraction] =
    useState<ReadingInteractionMenuSelection | null>(null);
  const [openInlineInsightsByTarget, setOpenInlineInsightsByTarget] =
    useState<InlineInsightStateByTarget>({});
  const [quizStateByTarget, setQuizStateByTarget] =
    useState<QuizStateByTarget>({});
  const [criticalThinkingStateByTarget, setCriticalThinkingStateByTarget] =
    useState<CriticalThinkingStateByTarget>({});
  const requestIdRef = useRef(0);
  const [generationTasks] = useState(() => new InteractionGenerationTasks());
  const activationRef = useRef<{ controller: AbortController; targetKey: string; kind: ReadingInteractionKind } | null>(null);
  useEffect(() => () => {
    activationRef.current?.controller.abort();
    generationTasks.dispose();
    requestIdRef.current += 1;
  }, [generationTasks]);

  function cancelActivation() {
    const active = activationRef.current;
    active?.controller.abort();
    activationRef.current = null;
    requestIdRef.current += 1;
    if (active?.kind === "insight") {
      setOpenInlineInsightsByTarget(current => {
        const insight = current[active.targetKey];
        if (!insight || !["loading", "generating"].includes(insight.status)) return current;
        return { ...current, [active.targetKey]: { ...insight, status: "idle" } };
      });
    }
  }

  async function activateInteraction(selection: ReadingInteractionMenuSelection,
    target: ReadingInteractionTargetRequest, service: ReadingInteractionService,
    onError: (message: string) => void, retryGeneration = false, onQuestionGenerated?: (key: string) => void) {
    cancelActivation();
    const activation = { controller: new AbortController(), targetKey: selection.targetKey, kind: selection.kind };
    activationRef.current = activation;
    const signal = activation.controller.signal;
    const key = selection.targetKey;
    const sourceEpoch = generationTasks.sourceEpoch(target.doc_name);
    // The same guard covers both phases, including notifications and failures.
    async function run<T extends { envelope: ReadingInteractionResponseEnvelope; payload?: unknown }, S>(
      interactionType: string, read: () => Promise<T>, generate: (signal: AbortSignal) => Promise<T>,
      begin: (key: string, status: "loading" | "generating") => number,
      map: (response: T) => S, apply: (key: string, state: S, id: number) => void,
      fail: (key: string, id: number, message: string, status: ReturnType<typeof normalizeReadingInteractionThrownError>["status"]) => void,
    ) {
      let requestId = begin(key, "loading");
      let generated = false;
      const isCurrent = () => !signal.aborted && generationTasks.sourceEpoch(target.doc_name) === sourceEpoch
        && activationRef.current === activation && isLatestInteractionRequest(requestId);
      try {
        const taskKey = JSON.stringify([interactionType, key]);
        const pending = generationTasks.find<T>(target.doc_name, taskKey, retryGeneration);
        const onGenerating = () => { generated = true; requestId = begin(key, "generating"); };
        if (pending) onGenerating();
        const response = await (pending ?? readOrGenerate({ target, interactionType, read,
          generate: () => generationTasks.start(target.doc_name, taskKey, async taskSignal =>
            validateInteractionResponse(await generate(taskSignal), target, interactionType)),
          isCurrent, retryGeneration, onGenerating }));
        if (response && isCurrent()) {
          apply(key, map(response), requestId);
          if (generated && selection.kind === "critical_thinking" && response.envelope.status === "question_generated") {
            onQuestionGenerated?.(key);
          }
        }
      } catch (error) {
        if (!isCurrent()) return;
        const failure = normalizeReadingInteractionThrownError(selection.target, error);
        fail(key, requestId, failure.message, failure.status);
        onError(failure.message);
      } finally {
        if (activationRef.current === activation) activationRef.current = null;
      }
    }
    if (selection.kind === "insight") {
      await run("analysis", () => service.readInsight(target, signal), taskSignal => service.generateInsight(target, { signal: taskSignal }),
        beginInlineInsightRequest, response => mapAnalysisInteractionResponseToInsightViewState(selection.target, response),
        applyInlineInsightState, failInlineInsightRequest);
    } else if (selection.kind === "quiz") {
      await run("quiz", () => service.readQuiz(target, signal), taskSignal => service.generateQuiz(target, { signal: taskSignal }),
        beginQuizRequest, response => mapQuizInteractionResponseToQuizViewState(selection.target, response), applyQuizState, failQuizRequest);
    } else if (selection.kind === "critical_thinking") {
      await run("critical_thinking_session", () => service.readCriticalThinkingSession(target, { signal }),
        taskSignal => service.generateCriticalThinkingQuestion(target, { signal: taskSignal }), beginCriticalThinkingRequest,
        response => mapCriticalThinkingSessionResponseToViewState(selection.target, response), applyCriticalThinkingState, failCriticalThinkingRequest);
    }
  }

  function openInteractionMenu(target: ReadingInteractionTarget) {
    setInteractionMenuTarget(target);
  }

  function closeInteractionMenu() {
    setInteractionMenuTarget(null);
  }

  function selectInteractionAction(
    kind: ReadingInteractionKind,
    targetOverride?: ReadingInteractionTarget,
  ): ReadingInteractionMenuSelection | null {
    const target = targetOverride || interactionMenuTarget;
    if (!target) {
      closeInteractionMenu();
      return null;
    }

    const active = activationRef.current;
    if (active && active.kind === kind && active.targetKey === buildReadingInteractionTargetKey(target)) {
      closeInteractionMenu();
      return null;
    }
    cancelActivation();

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const selection: ReadingInteractionMenuSelection = {
      kind,
      target,
      targetKey: buildReadingInteractionTargetKey(target),
      breadcrumb: buildReadingInteractionBreadcrumb(target),
      surface: surfaceForInteractionKind(kind),
      requestId,
    };
    setSelectedInteraction(selection);
    if (kind === "insight") {
      setOpenInlineInsightsByTarget((current) => ({
        ...current,
        [selection.targetKey]: createInitialInsightViewState(target, {
          status: "not_generated",
          expanded: true,
        }),
      }));
    } else if (kind === "quiz") {
      setQuizStateByTarget((current) => ({
        ...current,
        [selection.targetKey]:
          current[selection.targetKey] || createInitialQuizViewState(target),
      }));
    } else if (kind === "critical_thinking") {
      setCriticalThinkingStateByTarget((current) => ({
        ...current,
        [selection.targetKey]:
          current[selection.targetKey] || createInitialCriticalThinkingViewState(target),
      }));
    }
    closeInteractionMenu();
    return selection;
  }

  function isLatestInteractionRequest(requestId: number) {
    return requestIdRef.current === requestId;
  }

  function beginInlineInsightRequest(
    targetKey: ReadingInteractionTargetKey,
    status: Extract<InteractionStatus, "loading" | "generating" | "refreshing">,
  ) {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setOpenInlineInsightsByTarget((current) => {
      const insight = current[targetKey];
      if (!insight) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...insight,
          status,
          errorMessage: undefined,
        },
      };
    });
    return requestId;
  }

  function applyInlineInsightState(
    targetKey: ReadingInteractionTargetKey,
    nextInsight: InsightViewState,
    requestId: number,
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setOpenInlineInsightsByTarget((current) => ({
      ...current,
      [targetKey]: nextInsight,
    }));
  }

  function failInlineInsightRequest(
    targetKey: ReadingInteractionTargetKey,
    requestId: number,
    errorMessage: string,
    status: Extract<
      InteractionStatus,
      | "insufficient_content"
      | "stale_target"
      | "generation_failed"
      | "validation_failed"
      | "evaluation_failed"
    > = "validation_failed",
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setOpenInlineInsightsByTarget((current) => {
      const insight = current[targetKey];
      if (!insight) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...insight,
          status,
          errorMessage,
        },
      };
    });
  }

  function beginQuizRequest(
    targetKey: ReadingInteractionTargetKey,
    status: Extract<InteractionStatus, "loading" | "generating" | "refreshing">,
  ) {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setQuizStateByTarget((current) => {
      const quiz = current[targetKey];
      if (!quiz) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...quiz,
          status,
          errorMessage: undefined,
        },
      };
    });
    return requestId;
  }

  function applyQuizState(
    targetKey: ReadingInteractionTargetKey,
    nextQuiz: QuizViewState,
    requestId: number,
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setQuizStateByTarget((current) => ({
      ...current,
      [targetKey]: nextQuiz,
    }));
  }

  function failQuizRequest(
    targetKey: ReadingInteractionTargetKey,
    requestId: number,
    errorMessage: string,
    status: Extract<
      InteractionStatus,
      | "insufficient_content"
      | "stale_target"
      | "generation_failed"
      | "validation_failed"
      | "evaluation_failed"
    > = "validation_failed",
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setQuizStateByTarget((current) => {
      const quiz = current[targetKey];
      if (!quiz) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...quiz,
          status,
          errorMessage,
        },
      };
    });
  }

  function beginCriticalThinkingRequest(
    targetKey: ReadingInteractionTargetKey,
    status: "loading" | "generating" | "submitting" | "retrying",
  ) {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setCriticalThinkingStateByTarget((current) => {
      const session = current[targetKey];
      if (!session) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...session,
          status,
          errorMessage: undefined,
        },
      };
    });
    return requestId;
  }

  function applyCriticalThinkingState(
    targetKey: ReadingInteractionTargetKey,
    nextSession: CriticalThinkingViewState,
    requestId: number,
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setCriticalThinkingStateByTarget((current) => ({
      ...current,
      [targetKey]: nextSession,
    }));
  }

  function failCriticalThinkingRequest(
    targetKey: ReadingInteractionTargetKey,
    requestId: number,
    errorMessage: string,
    status: Extract<
      CriticalThinkingStatus,
      | "insufficient_content"
      | "stale_target"
      | "generation_failed"
      | "validation_failed"
      | "evaluation_failed"
    > = "validation_failed",
  ) {
    if (!isLatestInteractionRequest(requestId)) {
      return;
    }
    setCriticalThinkingStateByTarget((current) => {
      const session = current[targetKey];
      if (!session) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...session,
          status,
          errorMessage,
        },
      };
    });
  }

  function clearSelectedInteraction() {
    cancelActivation();
    setSelectedInteraction(null);
  }

  function invalidateInteractionSources(docName: string) {
    generationTasks.invalidateSources(docName);
  }

  /** Presentation reset: navigation detaches observers without losing paid work. */
  function resetInteractions() {
    cancelActivation();
    requestIdRef.current += 1;
    setInteractionMenuTarget(null);
    setSelectedInteraction(null);
    setOpenInlineInsightsByTarget({});
    setQuizStateByTarget({});
    setCriticalThinkingStateByTarget({});
  }

  function closeInlineInsight(targetKey: ReadingInteractionTargetKey) {
    if (activationRef.current?.kind === "insight" && activationRef.current.targetKey === targetKey) cancelActivation();
    setOpenInlineInsightsByTarget((current) => {
      const { [targetKey]: _removed, ...next } = current;
      return next;
    });
  }

  function toggleInlineInsightExpanded(targetKey: ReadingInteractionTargetKey) {
    setOpenInlineInsightsByTarget((current) => {
      const insight = current[targetKey];
      if (!insight) {
        return current;
      }
      return {
        ...current,
        [targetKey]: {
          ...insight,
          expanded: !insight.expanded,
        },
      };
    });
  }

  return {
    generationTasks,
    invalidateInteractionSources,
    activateInteraction,
    resetInteractions,
    interactionMenuTarget,
    selectedInteraction,
    openInlineInsightsByTarget,
    quizStateByTarget,
    criticalThinkingStateByTarget,
    openInteractionMenu,
    closeInteractionMenu,
    selectInteractionAction,
    isLatestInteractionRequest,
    beginInlineInsightRequest,
    applyInlineInsightState,
    failInlineInsightRequest,
    beginQuizRequest,
    applyQuizState,
    failQuizRequest,
    beginCriticalThinkingRequest,
    applyCriticalThinkingState,
    failCriticalThinkingRequest,
    clearSelectedInteraction,
    closeInlineInsight,
    toggleInlineInsightExpanded,
  };
}
