import type {
  AnalysisInteractionResponse,
  ReadingInteractionTargetRequest,
  ReadingInteractionTargetType,
} from "../../types/api";

export type ReadingInteractionTargetLevel =
  | "document"
  | "chapter"
  | "section"
  | "task_unit";

export type ReadingInteractionKind =
  | "insight"
  | "quiz"
  | "critical_thinking";

export interface ReadingInteractionTarget {
  targetLevel: ReadingInteractionTargetLevel;
  docName: string;
  documentId?: string;
  chapterId?: string;
  sectionId?: string;
  taskUnitId?: string;
  parentChapterId?: string;
  parentSectionId?: string;
  displayTitle: string;
  breadcrumb: string[];
  sourceStructureVersion?: string;
  sourceHash?: string;
}

export type ReadingInteractionTargetKey = string;

export type InteractionStatus =
  | "idle"
  | "loading"
  | "not_generated"
  | "generating"
  | "refreshing"
  | "submitting"
  | "retrying"
  | "completed"
  | "insufficient_content"
  | "stale_target"
  | "generation_failed"
  | "validation_failed"
  | "evaluation_failed";

export type CriticalThinkingStatus =
  | "idle"
  | "loading"
  | "not_generated"
  | "generating"
  | "question_generated"
  | "submitting"
  | "answer_submitted"
  | "retrying"
  | "completed"
  | "insufficient_content"
  | "stale_target"
  | "generation_failed"
  | "validation_failed"
  | "evaluation_failed";

export interface InteractionMetadataView {
  artifactId?: string;
  sessionId?: string;
  generatedAt?: string;
  updatedAt?: string;
  targetLevel: ReadingInteractionTargetLevel;
  referencedArtifactSummary?: Array<{
    artifactId: string;
    artifactType: string;
    targetLevel: ReadingInteractionTargetLevel;
  }>;
}

export interface InsightViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  content?: string;
  metadata?: InteractionMetadataView;
  expanded: boolean;
  errorMessage?: string;
}

export type QuizItemType = "short_answer" | "multiple_choice" | "true_false";

export interface QuizOptionView {
  optionId: string;
  label: string;
  text: string;
  isCorrect?: boolean;
}

export interface QuizItemView {
  itemId: string;
  itemType: QuizItemType;
  prompt: string;
  options?: QuizOptionView[];
  answer?: string;
  explanation?: string;
}

export type QuizPracticeAnswersByItem = Record<string, string>;
export type QuizPracticeRevealByItem = Record<string, boolean>;

export interface QuizViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  itemCount: number;
  items: QuizItemView[];
  generatedAt?: string;
  refreshedAt?: string;
  errorMessage?: string;
}

export interface CriticalThinkingEvaluationView {
  feedback?: string;
  score?: number;
  suggestedRefinement?: string;
}

export interface CriticalThinkingViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: CriticalThinkingStatus;
  question?: string;
  submittedAnswer?: string;
  evaluation?: CriticalThinkingEvaluationView;
  generatedAt?: string;
  evaluatedAt?: string;
  errorMessage?: string;
}

export type InlineInsightStateByTarget = Record<
  ReadingInteractionTargetKey,
  InsightViewState
>;

export type QuizStateByTarget = Record<
  ReadingInteractionTargetKey,
  QuizViewState
>;

export type CriticalThinkingStateByTarget = Record<
  ReadingInteractionTargetKey,
  CriticalThinkingViewState
>;

function normalizeKeySegment(value: string | undefined): string {
  return encodeURIComponent(value?.trim() || "");
}

function documentScopeForTarget(target: ReadingInteractionTarget): string {
  return target.documentId?.trim() || target.docName.trim();
}

export function hasRequiredTargetIdentity(target: ReadingInteractionTarget): boolean {
  if (!target.docName.trim()) {
    return false;
  }

  switch (target.targetLevel) {
    case "document":
      return true;
    case "chapter":
      return Boolean(target.chapterId?.trim());
    case "section":
      return Boolean(target.sectionId?.trim());
    case "task_unit":
      return Boolean(target.taskUnitId?.trim());
    default:
      return false;
  }
}

export function buildReadingInteractionTargetKey(
  target: ReadingInteractionTarget,
): ReadingInteractionTargetKey {
  if (!hasRequiredTargetIdentity(target)) {
    throw new Error("Reading interaction target is missing required backend identity.");
  }

  const documentScope = normalizeKeySegment(documentScopeForTarget(target));

  switch (target.targetLevel) {
    case "document":
      return `document:${documentScope}`;
    case "chapter":
      return `chapter:${documentScope}:${normalizeKeySegment(target.chapterId)}`;
    case "section":
      return [
        "section",
        documentScope,
        normalizeKeySegment(target.parentChapterId || target.chapterId),
        normalizeKeySegment(target.sectionId),
      ].join(":");
    case "task_unit":
      return [
        "task_unit",
        documentScope,
        normalizeKeySegment(target.parentChapterId || target.chapterId),
        normalizeKeySegment(target.parentSectionId || target.sectionId),
        normalizeKeySegment(target.taskUnitId),
      ].join(":");
    default:
      throw new Error("Unsupported reading interaction target level.");
  }
}

export function buildReadingInteractionBreadcrumb(
  target: ReadingInteractionTarget,
): string[] {
  const breadcrumb = target.breadcrumb
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  if (breadcrumb.length > 0) {
    return breadcrumb;
  }

  const displayTitle = target.displayTitle.trim();
  const fallbackTitle = displayTitle || target.docName.trim();

  return fallbackTitle ? [fallbackTitle] : [];
}

export function formatReadingInteractionBreadcrumb(
  target: ReadingInteractionTarget,
): string {
  return buildReadingInteractionBreadcrumb(target).join(" > ");
}

function normalizeSourceStructureVersion(
  sourceStructureVersion: string | undefined,
): number | null {
  if (!sourceStructureVersion?.trim()) {
    return null;
  }
  const parsed = Number(sourceStructureVersion);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

export function toReadingInteractionTargetRequest(
  target: ReadingInteractionTarget,
): ReadingInteractionTargetRequest {
  const targetType: ReadingInteractionTargetType = target.targetLevel;
  return {
    doc_name: target.docName,
    target_type: targetType,
    chapter_id:
      target.targetLevel === "chapter" ||
      target.targetLevel === "section" ||
      target.targetLevel === "task_unit"
        ? target.chapterId || target.parentChapterId || null
        : null,
    section_id:
      target.targetLevel === "section" || target.targetLevel === "task_unit"
        ? target.sectionId || target.parentSectionId || null
        : null,
    task_unit_id: target.targetLevel === "task_unit" ? target.taskUnitId || null : null,
    source_structure_version: normalizeSourceStructureVersion(
      target.sourceStructureVersion,
    ),
    source_hash: target.sourceHash || null,
  };
}

function normalizeInteractionStatus(status: string): InteractionStatus {
  const normalized = status.trim().toLowerCase().replace(/-/g, "_");
  const allowedStatuses: InteractionStatus[] = [
    "idle",
    "loading",
    "not_generated",
    "generating",
    "refreshing",
    "submitting",
    "retrying",
    "completed",
    "insufficient_content",
    "stale_target",
    "generation_failed",
    "validation_failed",
    "evaluation_failed",
  ];
  return allowedStatuses.includes(normalized as InteractionStatus)
    ? (normalized as InteractionStatus)
    : "validation_failed";
}

function analysisPayloadContent(response: AnalysisInteractionResponse): string | undefined {
  const payload = response.payload;
  if (!payload) {
    return undefined;
  }

  const segments = [
    payload.summary,
    payload.interpretation,
    payload.explanation,
    ...(payload.key_points || []).map((keyPoint) => `- ${keyPoint}`),
  ].filter((segment): segment is string => Boolean(segment?.trim()));

  return segments.join("\n\n");
}

export function mapAnalysisInteractionResponseToInsightViewState(
  target: ReadingInteractionTarget,
  response: AnalysisInteractionResponse,
  options: { expanded?: boolean } = {},
): InsightViewState {
  const status = normalizeInteractionStatus(response.envelope.status);
  const reason = response.envelope.reason?.trim();

  return createInitialInsightViewState(target, {
    status,
    expanded: options.expanded ?? true,
    content: status === "completed" ? analysisPayloadContent(response) : undefined,
    metadata: {
      artifactId: response.envelope.artifact_id || undefined,
      sessionId: response.envelope.session_id || undefined,
      generatedAt: response.envelope.generated_at || undefined,
      updatedAt: response.envelope.updated_at || undefined,
      targetLevel: target.targetLevel,
      referencedArtifactSummary:
        response.envelope.artifact_context_metadata?.referenced_artifact_ids?.map(
          (artifactId, index) => ({
            artifactId,
            artifactType:
              response.envelope.artifact_context_metadata
                ?.referenced_artifact_types?.[index] || "unknown",
            targetLevel:
              response.envelope.artifact_context_metadata
                ?.referenced_artifact_target_levels?.[index] || "document",
          }),
        ),
    },
    errorMessage:
      status === "generation_failed" ||
      status === "validation_failed" ||
      status === "stale_target"
        ? reason
        : undefined,
  });
}

export function createInitialInsightViewState(
  target: ReadingInteractionTarget,
  options: {
    status?: InteractionStatus;
    expanded?: boolean;
    content?: string;
    metadata?: InteractionMetadataView;
    errorMessage?: string;
  } = {},
): InsightViewState {
  return {
    targetKey: buildReadingInteractionTargetKey(target),
    target,
    status: options.status || "idle",
    content: options.content,
    metadata: options.metadata,
    expanded: options.expanded ?? true,
    errorMessage: options.errorMessage,
  };
}

export function createInitialQuizViewState(
  target: ReadingInteractionTarget,
  options: {
    status?: InteractionStatus;
    itemCount?: number;
    items?: QuizItemView[];
    generatedAt?: string;
    refreshedAt?: string;
    errorMessage?: string;
  } = {},
): QuizViewState {
  const items = options.items || [];

  return {
    targetKey: buildReadingInteractionTargetKey(target),
    target,
    status: options.status || "not_generated",
    itemCount: options.itemCount ?? items.length,
    items,
    generatedAt: options.generatedAt,
    refreshedAt: options.refreshedAt,
    errorMessage: options.errorMessage,
  };
}

export function createInitialCriticalThinkingViewState(
  target: ReadingInteractionTarget,
  options: {
    status?: CriticalThinkingStatus;
    question?: string;
    submittedAnswer?: string;
    evaluation?: CriticalThinkingEvaluationView;
    generatedAt?: string;
    evaluatedAt?: string;
    errorMessage?: string;
  } = {},
): CriticalThinkingViewState {
  return {
    targetKey: buildReadingInteractionTargetKey(target),
    target,
    status: options.status || "not_generated",
    question: options.question,
    submittedAnswer: options.submittedAnswer,
    evaluation: options.evaluation,
    generatedAt: options.generatedAt,
    evaluatedAt: options.evaluatedAt,
    errorMessage: options.errorMessage,
  };
}
