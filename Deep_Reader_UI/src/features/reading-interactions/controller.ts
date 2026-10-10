import { useRef, useState } from "react";
import {
  buildReadingInteractionBreadcrumb,
  buildReadingInteractionTargetKey,
  createInitialCriticalThinkingViewState,
  createInitialInsightViewState,
  createInitialQuizViewState,
  type CriticalThinkingStateByTarget,
  type InlineInsightStateByTarget,
  type InsightViewState,
  type InteractionStatus,
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
          status: "validation_failed",
          errorMessage,
        },
      };
    });
  }

  function clearSelectedInteraction() {
    setSelectedInteraction(null);
  }

  function closeInlineInsight(targetKey: ReadingInteractionTargetKey) {
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
    clearSelectedInteraction,
    closeInlineInsight,
    toggleInlineInsightExpanded,
  };
}
