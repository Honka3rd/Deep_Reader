import type {
  AnalysisInteractionResponse,
  CriticalThinkingSessionResponse,
  QuizArtifactItemResponse,
  QuizInteractionResponse,
  ReadingInteractionTargetRequest,
  ReadingInteractionTargetType,
} from "../../types/api";

export type ReadingInteractionTargetLevel =
  | "document"
  | "chapter"
  | "section"
  | "task_unit"
  | "content_block";

export type ReadingInteractionKind =
  | "insight"
  | "quiz"
  | "critical_thinking"
  | "summary"
  | "note";

export interface ReadingInteractionTarget {
  targetLevel: ReadingInteractionTargetLevel;
  docName: string;
  documentId?: string;
  chapterId?: string;
  sectionId?: string;
  taskUnitId?: string;
  contentBlockId?: string;
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

export type ReadingInteractionAvailabilityStatus =
  | InteractionStatus
  | CriticalThinkingStatus;

export type ReadingInteractionFailureStatus = Extract<
  InteractionStatus,
  | "insufficient_content"
  | "stale_target"
  | "generation_failed"
  | "validation_failed"
  | "evaluation_failed"
>;

export interface ReadingInteractionFailureView {
  status: ReadingInteractionFailureStatus;
  message: string;
}

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
  metadata?: InteractionMetadataView;
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
  metadata?: InteractionMetadataView;
  retryEligible?: boolean;
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
    case "content_block":
      return Boolean(target.taskUnitId?.trim() && target.contentBlockId?.trim());
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
    case "content_block":
      return [
        "content_block",
        documentScope,
        normalizeKeySegment(target.parentChapterId || target.chapterId),
        normalizeKeySegment(target.parentSectionId || target.sectionId),
        normalizeKeySegment(target.taskUnitId),
        normalizeKeySegment(target.contentBlockId),
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
      target.targetLevel === "task_unit" ||
      target.targetLevel === "content_block"
        ? target.chapterId || target.parentChapterId || null
        : null,
    section_id:
      target.targetLevel === "section" ||
      target.targetLevel === "task_unit" ||
      target.targetLevel === "content_block"
        ? target.sectionId || target.parentSectionId || null
        : null,
    task_unit_id:
      target.targetLevel === "task_unit" || target.targetLevel === "content_block"
        ? target.taskUnitId || null
        : null,
    content_block_id:
      target.targetLevel === "content_block" ? target.contentBlockId || null : null,
    source_structure_version: normalizeSourceStructureVersion(
      target.sourceStructureVersion,
    ),
    source_hash: target.sourceHash || null,
  };
}

export function normalizeInteractionStatus(status: string): InteractionStatus {
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

export function normalizeCriticalThinkingStatus(status: string): CriticalThinkingStatus {
  const normalized = status.trim().toLowerCase().replace(/-/g, "_");
  const allowedStatuses: CriticalThinkingStatus[] = [
    "idle",
    "loading",
    "not_generated",
    "generating",
    "question_generated",
    "submitting",
    "answer_submitted",
    "retrying",
    "completed",
    "insufficient_content",
    "stale_target",
    "generation_failed",
    "validation_failed",
    "evaluation_failed",
  ];
  return allowedStatuses.includes(normalized as CriticalThinkingStatus)
    ? (normalized as CriticalThinkingStatus)
    : "validation_failed";
}

export function normalizeReadingInteractionAvailabilityStatus(
  status: string,
): ReadingInteractionAvailabilityStatus {
  const normalized = status.trim().toLowerCase().replace(/-/g, "_");
  const allowedStatuses: ReadingInteractionAvailabilityStatus[] = [
    "idle",
    "loading",
    "not_generated",
    "generating",
    "refreshing",
    "submitting",
    "retrying",
    "question_generated",
    "answer_submitted",
    "completed",
    "insufficient_content",
    "stale_target",
    "generation_failed",
    "validation_failed",
    "evaluation_failed",
  ];
  return allowedStatuses.includes(normalized as ReadingInteractionAvailabilityStatus)
    ? (normalized as ReadingInteractionAvailabilityStatus)
    : "validation_failed";
}

export function formatReadingInteractionStatusLabel(
  status: ReadingInteractionAvailabilityStatus,
): string {
  switch (status) {
    case "not_generated":
      return "Not generated";
    case "question_generated":
      return "Question ready";
    case "answer_submitted":
      return "Answer submitted";
    case "insufficient_content":
      return "Insufficient content";
    case "stale_target":
      return "Stale target";
    case "generation_failed":
      return "Generation failed";
    case "validation_failed":
      return "Needs attention";
    case "evaluation_failed":
      return "Evaluation failed";
    case "completed":
      return "Ready";
    case "loading":
      return "Loading";
    case "generating":
      return "Generating";
    case "refreshing":
      return "Refreshing";
    case "submitting":
      return "Submitting";
    case "retrying":
      return "Retrying";
    case "idle":
    default:
      return "Available";
  }
}

function isContentBlockTarget(target: ReadingInteractionTarget): boolean {
  return target.targetLevel === "content_block";
}

function defaultInteractionErrorMessage(
  target: ReadingInteractionTarget,
  status: ReadingInteractionFailureStatus,
): string {
  const targetLabel = isContentBlockTarget(target)
    ? `content block "${target.displayTitle}"`
    : `target "${target.displayTitle}"`;

  switch (status) {
    case "insufficient_content":
      return `The selected ${targetLabel} does not have enough content for this interaction.`;
    case "stale_target":
      return `The selected ${targetLabel} is no longer available in the current document structure. Reopen the document content before retrying this same block.`;
    case "generation_failed":
      return `Generation failed for the selected ${targetLabel}.`;
    case "evaluation_failed":
      return `Evaluation failed for the selected ${targetLabel}.`;
    case "validation_failed":
    default:
      return `The selected ${targetLabel} could not be used for this interaction.`;
  }
}

export function normalizeReadingInteractionFailure(
  target: ReadingInteractionTarget,
  status: ReadingInteractionAvailabilityStatus,
  reason?: string,
): ReadingInteractionFailureView | undefined {
  const normalizedReason = reason?.trim();
  if (
    status !== "insufficient_content" &&
    status !== "stale_target" &&
    status !== "generation_failed" &&
    status !== "validation_failed" &&
    status !== "evaluation_failed"
  ) {
    return undefined;
  }

  return {
    status,
    message: normalizedReason || defaultInteractionErrorMessage(target, status),
  };
}

export function normalizeReadingInteractionThrownError(
  target: ReadingInteractionTarget,
  error: unknown,
): ReadingInteractionFailureView {
  const rawMessage = error instanceof Error ? error.message : String(error);
  const message = rawMessage.trim();
  const normalized = message.toLowerCase().replace(/-/g, "_");
  let status: ReadingInteractionFailureStatus = "validation_failed";

  if (
    normalized.includes("insufficient_content") ||
    normalized.includes("too short") ||
    normalized.includes("not enough content")
  ) {
    status = "insufficient_content";
  } else if (
    normalized.includes("stale") ||
    normalized.includes("source_hash") ||
    normalized.includes("source_structure_version")
  ) {
    status = "stale_target";
  } else if (
    isContentBlockTarget(target) &&
    (normalized.includes("not_found") ||
      normalized.includes("not found") ||
      normalized.includes("missing content_block") ||
      normalized.includes("content_block_id") ||
      normalized.includes("resolver"))
  ) {
    status = "stale_target";
  } else if (normalized.includes("generation_failed")) {
    status = "generation_failed";
  } else if (normalized.includes("evaluation_failed")) {
    status = "evaluation_failed";
  }

  return {
    status,
    message: message || defaultInteractionErrorMessage(target, status),
  };
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

function mapInteractionMetadata(
  target: ReadingInteractionTarget,
  response:
    | AnalysisInteractionResponse
    | QuizInteractionResponse
    | CriticalThinkingSessionResponse,
): InteractionMetadataView {
  return {
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
  };
}

export function mapAnalysisInteractionResponseToInsightViewState(
  target: ReadingInteractionTarget,
  response: AnalysisInteractionResponse,
  options: { expanded?: boolean } = {},
): InsightViewState {
  const status = normalizeInteractionStatus(response.envelope.status);
  const failure = normalizeReadingInteractionFailure(
    target,
    status,
    response.envelope.reason || undefined,
  );

  return createInitialInsightViewState(target, {
    status,
    expanded: options.expanded ?? true,
    content: status === "completed" ? analysisPayloadContent(response) : undefined,
    metadata: mapInteractionMetadata(target, response),
    errorMessage: failure?.message,
  });
}

function normalizeQuizItemType(itemType: string): QuizItemType {
  const normalized = itemType.trim().toLowerCase().replace(/-/g, "_");
  if (
    normalized === "short_answer" ||
    normalized === "multiple_choice" ||
    normalized === "true_false"
  ) {
    return normalized;
  }
  return "short_answer";
}

function quizOptionLabel(index: number): string {
  const alphabetCode = "A".charCodeAt(0) + index;
  return alphabetCode <= "Z".charCodeAt(0)
    ? String.fromCharCode(alphabetCode)
    : String(index + 1);
}

function quizAnswerToText(answer: string | boolean): string {
  return typeof answer === "boolean" ? (answer ? "True" : "False") : answer;
}

function mapQuizItemResponseToView(
  item: QuizArtifactItemResponse,
  index: number,
): QuizItemView {
  const itemId = item.item_id.trim() || `q${index + 1}`;
  return {
    itemId,
    itemType: normalizeQuizItemType(item.item_type),
    prompt: item.prompt.trim(),
    options: item.options?.map((option, optionIndex) => ({
      optionId: `${itemId}:option:${optionIndex + 1}`,
      label: quizOptionLabel(optionIndex),
      text: option.trim(),
      isCorrect:
        typeof item.answer === "string"
          ? option.trim() === item.answer.trim()
          : undefined,
    })),
    answer: quizAnswerToText(item.answer),
    explanation: item.explanation?.trim() || undefined,
  };
}

export function mapQuizInteractionResponseToQuizViewState(
  target: ReadingInteractionTarget,
  response: QuizInteractionResponse,
): QuizViewState {
  const status = normalizeInteractionStatus(response.envelope.status);
  const failure = normalizeReadingInteractionFailure(
    target,
    status,
    response.envelope.reason || undefined,
  );
  const items =
    status === "completed"
      ? response.payload?.items.map(mapQuizItemResponseToView) || []
      : [];

  return createInitialQuizViewState(target, {
    status,
    itemCount: items.length,
    items,
    metadata: mapInteractionMetadata(target, response),
    generatedAt: response.envelope.generated_at || undefined,
    refreshedAt: response.envelope.updated_at || undefined,
    errorMessage: failure?.message,
  });
}

function mapCriticalThinkingEvaluation(
  response: CriticalThinkingSessionResponse,
): CriticalThinkingEvaluationView | undefined {
  const evaluation = response.payload?.evaluation;
  if (!evaluation) {
    return undefined;
  }

  return {
    feedback: evaluation.feedback,
    score: evaluation.score ?? undefined,
    suggestedRefinement: evaluation.suggested_refinement || undefined,
  };
}

export function mapCriticalThinkingSessionResponseToViewState(
  target: ReadingInteractionTarget,
  response: CriticalThinkingSessionResponse,
): CriticalThinkingViewState {
  const status = normalizeCriticalThinkingStatus(response.envelope.status);
  const failure = normalizeReadingInteractionFailure(
    target,
    status,
    response.envelope.reason || undefined,
  );
  const payload = response.payload;
  const hasEvaluationTimestamp =
    status === "completed" || status === "evaluation_failed";

  return createInitialCriticalThinkingViewState(target, {
    status,
    question: payload?.question?.trim() || undefined,
    submittedAnswer: payload?.submitted_answer?.trim() || undefined,
    evaluation: mapCriticalThinkingEvaluation(response),
    metadata: mapInteractionMetadata(target, response),
    retryEligible: payload?.retry_eligible ?? status === "evaluation_failed",
    generatedAt: response.envelope.generated_at || undefined,
    evaluatedAt: hasEvaluationTimestamp
      ? response.envelope.updated_at || undefined
      : undefined,
    errorMessage: failure?.message,
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
    metadata?: InteractionMetadataView;
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
    metadata: options.metadata,
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
    metadata?: InteractionMetadataView;
    retryEligible?: boolean;
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
    metadata: options.metadata,
    retryEligible: options.retryEligible,
    generatedAt: options.generatedAt,
    evaluatedAt: options.evaluatedAt,
    errorMessage: options.errorMessage,
  };
}
