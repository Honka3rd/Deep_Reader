import {
  Box,
  Button,
  Drawer,
  FormControlLabel,
  Radio,
  RadioGroup,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useEffect, useState, type ReactNode } from "react";
import type { ReadingInteractionMenuSelection } from "./controller";
import type {
  CriticalThinkingStatus,
  CriticalThinkingViewState,
  InsightViewState,
  InteractionStatus,
  QuizPracticeAnswersByItem,
  QuizPracticeRevealByItem,
  QuizItemType,
  QuizItemView,
  QuizViewState,
} from "./model";

export function ReadingInteractionsView({
  drawerSelection,
  commandSlot,
  contentSlot,
  errorMessage,
  loading = false,
  onCloseDrawer,
}: {
  drawerSelection: ReadingInteractionMenuSelection | null;
  commandSlot?: ReactNode;
  contentSlot?: ReactNode;
  errorMessage?: string;
  loading?: boolean;
  onCloseDrawer: () => void;
}) {
  const open = Boolean(drawerSelection);
  const drawerTitle =
    drawerSelection?.kind === "critical_thinking"
      ? "Critical thinking"
      : drawerSelection?.kind === "quiz"
      ? "Quiz"
      : "Reading interaction";

  return (
    <Drawer
      anchor="right"
      className="reading-interaction-drawer"
      open={open}
      onClose={onCloseDrawer}
      PaperProps={{ className: "reading-interaction-drawer-paper" }}
    >
      <Box className="reading-interaction-drawer-shell" role="region" aria-label={drawerTitle}>
        <Stack spacing={2}>
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            justifyContent="space-between"
            className="reading-interaction-drawer-header"
          >
            <Box className="reading-interaction-drawer-title-group">
              <Typography component="h2" variant="h2">
                {drawerTitle}
              </Typography>
              <Typography
                className="reading-interaction-drawer-breadcrumb"
                color="text.secondary"
                variant="body2"
              >
                {drawerSelection?.breadcrumb.join(" > ") || "No target"}
              </Typography>
            </Box>
            <Button
              className="reading-interaction-drawer-close"
              size="small"
              onClick={onCloseDrawer}
            >
              Close
            </Button>
          </Stack>
          <Box className="reading-interaction-drawer-command-slot">
            {commandSlot || (
              <Typography color="text.secondary" variant="body2">
                Commands will appear here when backend interaction routes are available.
              </Typography>
            )}
          </Box>
          <Box className="reading-interaction-drawer-body">
            {loading ? (
              <Typography color="text.secondary" variant="body2">
                Loading interaction.
              </Typography>
            ) : errorMessage ? (
              <Typography color="error" variant="body2">
                {errorMessage}
              </Typography>
            ) : contentSlot ? (
              contentSlot
            ) : (
              <Typography color="text.secondary" variant="body2">
                No interaction content has been loaded. Opening this drawer does not call
                generation or refresh APIs.
              </Typography>
            )}
          </Box>
        </Stack>
      </Box>
    </Drawer>
  );
}

export function CriticalThinkingDrawerCommands({
  session,
  onGenerateQuestion,
  onRetryEvaluation,
  onSubmitAnswer,
}: {
  session: CriticalThinkingViewState;
  onGenerateQuestion?: (targetKey: string) => void;
  onRetryEvaluation?: (targetKey: string) => void;
  onSubmitAnswer?: (targetKey: string) => void;
}) {
  const statusView = getCriticalThinkingStatusView(session);

  return (
    <Stack spacing={1.25} className="critical-thinking-command-panel">
      <Typography className="critical-thinking-command-title" variant="subtitle2">
        Critical-thinking session
      </Typography>
      <Typography color="text.secondary" variant="body2">
        Read-first placeholder. Opening this drawer does not generate a question or
        submit an answer.
      </Typography>
      {statusView.showCommands ? (
        <Stack direction="row" spacing={1} className="critical-thinking-actions">
          {statusView.canGenerateQuestion && onGenerateQuestion ? (
            <Button
              className="critical-thinking-generate"
              size="small"
              variant="contained"
              onClick={() => onGenerateQuestion(session.targetKey)}
            >
              Generate question
            </Button>
          ) : null}
          {statusView.canSubmitAnswer && onSubmitAnswer ? (
            <Button
              className="critical-thinking-submit"
              size="small"
              variant="outlined"
              onClick={() => onSubmitAnswer(session.targetKey)}
            >
              Submit answer
            </Button>
          ) : null}
          {statusView.canRetryEvaluation && onRetryEvaluation ? (
            <Button
              className="critical-thinking-retry"
              size="small"
              variant="outlined"
              onClick={() => onRetryEvaluation(session.targetKey)}
            >
              Retry evaluation
            </Button>
          ) : null}
        </Stack>
      ) : null}
    </Stack>
  );
}

export function CriticalThinkingDrawerContent({
  onDraftDirtyChange,
  session,
}: {
  onDraftDirtyChange?: (dirty: boolean) => void;
  session: CriticalThinkingViewState;
}) {
  const statusView = getCriticalThinkingStatusView(session);
  const [draftAnswer, setDraftAnswer] = useState("");
  const hasQuestion = Boolean(session.question?.trim());
  const hasEvaluation =
    Boolean(session.evaluation?.feedback?.trim()) ||
    typeof session.evaluation?.score === "number" ||
    Boolean(session.evaluation?.suggestedRefinement?.trim());

  useEffect(() => {
    setDraftAnswer("");
    onDraftDirtyChange?.(false);
  }, [onDraftDirtyChange, session.targetKey, session.question, session.submittedAnswer]);

  function updateDraftAnswer(nextDraftAnswer: string) {
    setDraftAnswer(nextDraftAnswer);
    onDraftDirtyChange?.(nextDraftAnswer.trim().length > 0);
  }

  return (
    <Stack
      spacing={1.5}
      className={[
        "critical-thinking-content",
        `critical-thinking-status-${session.status}`,
      ].join(" ")}
      data-target-key={session.targetKey}
    >
      <Box className="critical-thinking-status-panel">
        <Typography className="critical-thinking-status-label" variant="caption">
          {statusView.label}
        </Typography>
        <Typography className="critical-thinking-status-message" variant="body2">
          {statusView.message}
        </Typography>
      </Box>
      {session.errorMessage ? (
        <Typography className="critical-thinking-error" color="error" variant="body2">
          {session.errorMessage}
        </Typography>
      ) : null}
      {hasQuestion ? (
        <Box className="critical-thinking-question-panel">
          <Typography className="critical-thinking-section-title" variant="subtitle2">
            Question
          </Typography>
          <Typography className="critical-thinking-question" variant="body2">
            {session.question}
          </Typography>
        </Box>
      ) : null}
      {session.status === "question_generated" || session.submittedAnswer ? (
        <Box className="critical-thinking-answer-panel">
          <Typography className="critical-thinking-section-title" variant="subtitle2">
            Answer
          </Typography>
          {session.submittedAnswer ? (
            <Typography className="critical-thinking-submitted-answer" variant="body2">
              {session.submittedAnswer}
            </Typography>
          ) : (
            <TextField
              className="critical-thinking-answer-input"
              label="Draft answer"
              minRows={5}
              multiline
              placeholder="Write your answer"
              size="small"
              value={draftAnswer}
              onChange={(event) => updateDraftAnswer(event.target.value)}
            />
          )}
        </Box>
      ) : null}
      {hasEvaluation ? (
        <Box className="critical-thinking-evaluation-panel">
          <Typography className="critical-thinking-section-title" variant="subtitle2">
            Evaluation
          </Typography>
          {typeof session.evaluation?.score === "number" ? (
            <Typography className="critical-thinking-score" variant="body2">
              Score: {session.evaluation.score}
            </Typography>
          ) : null}
          {session.evaluation?.feedback ? (
            <Typography className="critical-thinking-feedback" variant="body2">
              {session.evaluation.feedback}
            </Typography>
          ) : null}
          {session.evaluation?.suggestedRefinement ? (
            <Typography
              className="critical-thinking-suggested-refinement"
              color="text.secondary"
              variant="body2"
            >
              {session.evaluation.suggestedRefinement}
            </Typography>
          ) : null}
        </Box>
      ) : null}
      {session.generatedAt || session.evaluatedAt ? (
        <Stack spacing={0.25} className="critical-thinking-metadata">
          {session.generatedAt ? (
            <Typography color="text.secondary" variant="caption">
              Generated: {session.generatedAt}
            </Typography>
          ) : null}
          {session.evaluatedAt ? (
            <Typography color="text.secondary" variant="caption">
              Evaluated: {session.evaluatedAt}
            </Typography>
          ) : null}
        </Stack>
      ) : null}
    </Stack>
  );
}

export function QuizDrawerCommands({
  quiz,
  onGenerate,
  onRefresh,
}: {
  quiz: QuizViewState;
  onGenerate?: (targetKey: string) => void;
  onRefresh?: (targetKey: string) => void;
}) {
  const statusView = getQuizStatusView(quiz);

  return (
    <Stack spacing={1.25} className="quiz-drawer-command-panel">
      <Typography className="quiz-drawer-command-title" variant="subtitle2">
        Quiz artifact
      </Typography>
      <Typography color="text.secondary" variant="body2">
        Read-first placeholder. Opening this drawer does not generate or refresh a quiz.
      </Typography>
      {statusView.showCommands ? (
        <Stack direction="row" spacing={1} className="quiz-drawer-actions">
          {statusView.canGenerate && onGenerate ? (
            <Button
              className="quiz-drawer-generate"
              size="small"
              variant="contained"
              onClick={() => onGenerate(quiz.targetKey)}
            >
              Generate quiz
            </Button>
          ) : null}
          {statusView.canRefresh && onRefresh ? (
            <Button
              className="quiz-drawer-refresh"
              size="small"
              variant="outlined"
              onClick={() => onRefresh(quiz.targetKey)}
            >
              Refresh quiz
            </Button>
          ) : null}
        </Stack>
      ) : null}
    </Stack>
  );
}

export function QuizDrawerContent({ quiz }: { quiz: QuizViewState }) {
  const statusView = getQuizStatusView(quiz);
  const [practiceAnswersByItem, setPracticeAnswersByItem] =
    useState<QuizPracticeAnswersByItem>({});
  const [revealedItemsByItem, setRevealedItemsByItem] =
    useState<QuizPracticeRevealByItem>({});
  const itemSignature = quiz.items.map((item) => item.itemId).join("|");

  useEffect(() => {
    setPracticeAnswersByItem({});
    setRevealedItemsByItem({});
  }, [quiz.targetKey, itemSignature]);

  function updatePracticeAnswer(itemId: string, answer: string) {
    setPracticeAnswersByItem((current) => ({
      ...current,
      [itemId]: answer,
    }));
  }

  function toggleAnswerReveal(itemId: string) {
    setRevealedItemsByItem((current) => ({
      ...current,
      [itemId]: !current[itemId],
    }));
  }

  return (
    <Stack
      spacing={1.5}
      className={[
        "quiz-drawer-content",
        `quiz-drawer-status-${quiz.status}`,
      ].join(" ")}
      data-target-key={quiz.targetKey}
    >
      <Box className="quiz-drawer-status-panel">
        <Typography className="quiz-drawer-status-label" variant="caption">
          {statusView.label}
        </Typography>
        <Typography className="quiz-drawer-status-message" variant="body2">
          {statusView.message}
        </Typography>
      </Box>
      {quiz.errorMessage ? (
        <Typography className="quiz-drawer-error" color="error" variant="body2">
          {quiz.errorMessage}
        </Typography>
      ) : null}
      {quiz.status === "completed" ? (
        <Box className="quiz-drawer-item-summary">
          <Typography variant="subtitle2">Quiz items</Typography>
          {quiz.items.length > 0 ? (
            <QuizItemList
              answersByItem={practiceAnswersByItem}
              items={quiz.items}
              revealedItemsByItem={revealedItemsByItem}
              onChangeAnswer={updatePracticeAnswer}
              onToggleReveal={toggleAnswerReveal}
            />
          ) : (
            <Typography color="text.secondary" variant="body2">
              {quiz.itemCount > 0
                ? `${quiz.itemCount} item${quiz.itemCount === 1 ? "" : "s"} available, waiting for item payload.`
                : "No quiz items are available yet."}
            </Typography>
          )}
        </Box>
      ) : null}
      {quiz.generatedAt || quiz.refreshedAt ? (
        <Stack spacing={0.25} className="quiz-drawer-metadata">
          {quiz.generatedAt ? (
            <Typography color="text.secondary" variant="caption">
              Generated: {quiz.generatedAt}
            </Typography>
          ) : null}
          {quiz.refreshedAt ? (
            <Typography color="text.secondary" variant="caption">
              Refreshed: {quiz.refreshedAt}
            </Typography>
          ) : null}
        </Stack>
      ) : null}
    </Stack>
  );
}

export function QuizItemList({
  answersByItem,
  items,
  revealedItemsByItem,
  onChangeAnswer,
  onToggleReveal,
}: {
  answersByItem?: QuizPracticeAnswersByItem;
  items: QuizItemView[];
  revealedItemsByItem?: QuizPracticeRevealByItem;
  onChangeAnswer?: (itemId: string, answer: string) => void;
  onToggleReveal?: (itemId: string) => void;
}) {
  return (
    <Stack spacing={1.25} className="quiz-item-list">
      {items.map((item, index) => (
        <QuizItemCard
          answer={answersByItem?.[item.itemId] || ""}
          item={item}
          index={index}
          key={item.itemId || index}
          revealed={Boolean(revealedItemsByItem?.[item.itemId])}
          onChangeAnswer={onChangeAnswer}
          onToggleReveal={onToggleReveal}
        />
      ))}
    </Stack>
  );
}

function QuizItemCard({
  answer,
  item,
  index,
  revealed,
  onChangeAnswer,
  onToggleReveal,
}: {
  answer: string;
  item: QuizItemView;
  index: number;
  revealed: boolean;
  onChangeAnswer?: (itemId: string, answer: string) => void;
  onToggleReveal?: (itemId: string) => void;
}) {
  const typeLabel = quizItemTypeLabel(item.itemType);

  return (
    <Box className={`quiz-item-card quiz-item-card-${item.itemType}`}>
      <Stack spacing={1}>
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          justifyContent="space-between"
          className="quiz-item-header"
        >
          <Typography className="quiz-item-number" variant="caption">
            Question {index + 1}
          </Typography>
          <Typography className="quiz-item-type" variant="caption">
            {typeLabel}
          </Typography>
        </Stack>
        <Typography className="quiz-item-prompt" variant="body2">
          {item.prompt}
        </Typography>
        <QuizPracticeAnswerInput
          answer={answer}
          item={item}
          onChangeAnswer={onChangeAnswer}
        />
        <QuizItemOptions item={item} />
        <Stack direction="row" spacing={1} className="quiz-item-practice-actions">
          <Button
            className="quiz-item-reveal-answer"
            disabled={!item.answer && !item.explanation}
            size="small"
            variant="outlined"
            onClick={() => onToggleReveal?.(item.itemId)}
          >
            {revealed ? "Hide answer" : "Reveal answer"}
          </Button>
        </Stack>
        {revealed ? <QuizItemAnswer item={item} /> : null}
      </Stack>
    </Box>
  );
}

function QuizPracticeAnswerInput({
  answer,
  item,
  onChangeAnswer,
}: {
  answer: string;
  item: QuizItemView;
  onChangeAnswer?: (itemId: string, answer: string) => void;
}) {
  if (item.itemType === "short_answer") {
    return (
      <TextField
        className="quiz-item-practice-answer quiz-item-practice-answer-short"
        label="Practice answer"
        minRows={2}
        multiline
        size="small"
        value={answer}
        onChange={(event) => onChangeAnswer?.(item.itemId, event.target.value)}
      />
    );
  }

  const options =
    item.options && item.options.length > 0
      ? item.options
      : item.itemType === "true_false"
      ? [
          { optionId: "true", label: "True", text: "True" },
          { optionId: "false", label: "False", text: "False" },
        ]
      : [];

  if (options.length === 0) {
    return null;
  }

  return (
    <RadioGroup
      className="quiz-item-practice-answer quiz-item-practice-answer-choice"
      value={answer}
      onChange={(event) => onChangeAnswer?.(item.itemId, event.target.value)}
    >
      {options.map((option) => (
        <FormControlLabel
          className="quiz-item-practice-option"
          control={<Radio size="small" />}
          key={option.optionId}
          label={`${option.label}. ${option.text}`}
          value={option.optionId}
        />
      ))}
    </RadioGroup>
  );
}

function QuizItemOptions({ item }: { item: QuizItemView }) {
  if (item.itemType === "short_answer") {
    return null;
  }

  const options =
    item.options && item.options.length > 0
      ? item.options
      : item.itemType === "true_false"
      ? [
          { optionId: "true", label: "True", text: "True" },
          { optionId: "false", label: "False", text: "False" },
        ]
      : [];

  if (options.length === 0) {
    return (
      <Typography color="text.secondary" variant="body2">
        Options are not available for this question.
      </Typography>
    );
  }

  return (
    <Box component="ol" className="quiz-item-options">
      {options.map((option) => (
        <Box component="li" className="quiz-item-option" key={option.optionId}>
          <Typography variant="body2">
            <span className="quiz-item-option-label">{option.label}.</span> {option.text}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

function QuizItemAnswer({ item }: { item: QuizItemView }) {
  if (!item.answer && !item.explanation) {
    return null;
  }

  return (
    <Box className="quiz-item-answer-reveal">
      <Stack spacing={0.5}>
        <Typography className="quiz-item-answer-title" variant="caption">
          Answer
        </Typography>
        {item.answer ? (
          <Typography className="quiz-item-answer" variant="body2">
            {item.answer}
          </Typography>
        ) : null}
        {item.explanation ? (
          <Typography className="quiz-item-explanation" color="text.secondary" variant="body2">
            {item.explanation}
          </Typography>
        ) : null}
      </Stack>
    </Box>
  );
}

function quizItemTypeLabel(itemType: QuizItemType) {
  switch (itemType) {
    case "short_answer":
      return "Short answer";
    case "multiple_choice":
      return "Multiple choice";
    case "true_false":
      return "True / false";
    default:
      return "Quiz item";
  }
}

export function InlineInsightRegion({
  insight,
  className = "",
  onDismiss,
  onGenerate,
  onRefresh,
}: {
  insight: InsightViewState;
  className?: string;
  onDismiss?: (targetKey: string) => void;
  onGenerate?: (targetKey: string) => void;
  onRefresh?: (targetKey: string) => void;
}) {
  if (!insight.expanded) {
    return null;
  }

  const statusView = getInsightStatusView(insight);

  return (
    <Box
      className={[
        "inline-insight-region",
        `inline-insight-status-${insight.status}`,
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      data-target-key={insight.targetKey}
    >
      <Stack spacing={0.75}>
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          justifyContent="space-between"
          className="inline-insight-header"
        >
          <Typography className="inline-insight-title" variant="subtitle2">
            Insight
          </Typography>
          {onDismiss ? (
            <Button
              className="inline-insight-dismiss"
              size="small"
              onClick={() => onDismiss(insight.targetKey)}
            >
              Dismiss
            </Button>
          ) : null}
        </Stack>
        <Typography className="inline-insight-status-label" variant="caption">
          {statusView.label}
        </Typography>
        {insight.content ? (
          <Typography className="inline-insight-content" variant="body2">
            {insight.content}
          </Typography>
        ) : (
          <Typography
            className="inline-insight-placeholder"
            color="text.secondary"
            variant="body2"
          >
            {statusView.message}
          </Typography>
        )}
        {insight.errorMessage ? (
          <Typography className="inline-insight-error" color="error" variant="body2">
            {insight.errorMessage}
          </Typography>
        ) : null}
        {statusView.showCommands ? (
          <Stack direction="row" spacing={1} className="inline-insight-actions">
            {statusView.canGenerate && onGenerate ? (
              <Button
                className="inline-insight-generate"
                size="small"
                variant="contained"
                onClick={() => onGenerate(insight.targetKey)}
              >
                Generate
              </Button>
            ) : null}
            {statusView.canRefresh && onRefresh ? (
              <Button
                className="inline-insight-refresh"
                size="small"
                variant="outlined"
                onClick={() => onRefresh(insight.targetKey)}
              >
                Refresh
              </Button>
            ) : null}
          </Stack>
        ) : null}
      </Stack>
    </Box>
  );
}

function getCriticalThinkingStatusView(session: CriticalThinkingViewState): {
  label: string;
  message: string;
  canGenerateQuestion: boolean;
  canRetryEvaluation: boolean;
  canSubmitAnswer: boolean;
  showCommands: boolean;
} {
  const statusMessages: Record<CriticalThinkingStatus, { label: string; message: string }> = {
    idle: {
      label: "Ready",
      message: `Critical-thinking session is ready to load for ${session.target.displayTitle}.`,
    },
    loading: {
      label: "Loading",
      message: "Loading existing critical-thinking session.",
    },
    not_generated: {
      label: "Not generated",
      message: `No critical-thinking question exists yet for ${session.target.displayTitle}.`,
    },
    generating: {
      label: "Generating",
      message: "Generating critical-thinking question.",
    },
    question_generated: {
      label: "Question generated",
      message: "A question is available for local answer drafting.",
    },
    submitting: {
      label: "Submitting",
      message: "Submitting answer for evaluation.",
    },
    answer_submitted: {
      label: "Answer submitted",
      message: "The submitted answer is waiting for evaluation feedback.",
    },
    retrying: {
      label: "Retrying",
      message: "Retrying evaluation.",
    },
    completed: {
      label: "Completed",
      message: "Evaluation feedback is available.",
    },
    insufficient_content: {
      label: "Insufficient content",
      message: "This target does not currently have enough content for critical thinking.",
    },
    stale_target: {
      label: "Stale target",
      message: "This session may no longer match the current document structure.",
    },
    generation_failed: {
      label: "Generation failed",
      message: "Critical-thinking question generation failed.",
    },
    validation_failed: {
      label: "Validation failed",
      message: "Critical-thinking response validation failed.",
    },
    evaluation_failed: {
      label: "Evaluation failed",
      message: "Evaluation failed. Retry behavior will be wired in the next task.",
    },
  };
  const canGenerateQuestion = [
    "idle",
    "not_generated",
    "generation_failed",
    "validation_failed",
  ].includes(session.status);
  const canSubmitAnswer = session.status === "question_generated";
  const canRetryEvaluation = session.status === "evaluation_failed";

  return {
    ...statusMessages[session.status],
    canGenerateQuestion,
    canRetryEvaluation,
    canSubmitAnswer,
    showCommands: canGenerateQuestion || canSubmitAnswer || canRetryEvaluation,
  };
}

function getQuizStatusView(quiz: QuizViewState): {
  label: string;
  message: string;
  canGenerate: boolean;
  canRefresh: boolean;
  showCommands: boolean;
} {
  const statusMessages: Record<InteractionStatus, { label: string; message: string }> = {
    idle: {
      label: "Ready",
      message: `Quiz is ready to load for ${quiz.target.displayTitle}.`,
    },
    loading: {
      label: "Loading",
      message: "Loading existing quiz artifact.",
    },
    not_generated: {
      label: "Not generated",
      message: `No quiz exists yet for ${quiz.target.displayTitle}.`,
    },
    generating: {
      label: "Generating",
      message: "Generating quiz.",
    },
    refreshing: {
      label: "Refreshing",
      message: "Refreshing quiz.",
    },
    submitting: {
      label: "Submitting",
      message: "Submitting quiz interaction.",
    },
    retrying: {
      label: "Retrying",
      message: "Retrying quiz interaction.",
    },
    completed: {
      label: "Completed",
      message: "Quiz is available for this target.",
    },
    insufficient_content: {
      label: "Insufficient content",
      message: "This target does not currently have enough content for a quiz.",
    },
    stale_target: {
      label: "Stale target",
      message: "This quiz may no longer match the current document structure.",
    },
    generation_failed: {
      label: "Generation failed",
      message: "Quiz generation failed.",
    },
    validation_failed: {
      label: "Validation failed",
      message: "Quiz response validation failed.",
    },
    evaluation_failed: {
      label: "Evaluation failed",
      message: "Quiz evaluation failed.",
    },
  };

  const canGenerate = ["idle", "not_generated", "generation_failed", "validation_failed"].includes(
    quiz.status,
  );
  const canRefresh = ["completed", "stale_target", "generation_failed", "validation_failed"].includes(
    quiz.status,
  );

  return {
    ...statusMessages[quiz.status],
    canGenerate,
    canRefresh,
    showCommands: canGenerate || canRefresh,
  };
}

function getInsightStatusView(insight: InsightViewState): {
  label: string;
  message: string;
  canGenerate: boolean;
  canRefresh: boolean;
  showCommands: boolean;
} {
  const statusMessages: Record<InteractionStatus, { label: string; message: string }> = {
    idle: {
      label: "Ready",
      message: `Insight is ready to load for ${insight.target.displayTitle}.`,
    },
    loading: {
      label: "Loading",
      message: "Loading existing insight.",
    },
    not_generated: {
      label: "Not generated",
      message: `No insight exists yet for ${insight.target.displayTitle}.`,
    },
    generating: {
      label: "Generating",
      message: "Generating insight.",
    },
    refreshing: {
      label: "Refreshing",
      message: "Refreshing insight.",
    },
    submitting: {
      label: "Submitting",
      message: "Submitting interaction.",
    },
    retrying: {
      label: "Retrying",
      message: "Retrying interaction.",
    },
    completed: {
      label: "Completed",
      message: "Insight is available.",
    },
    insufficient_content: {
      label: "Insufficient content",
      message: "This target does not currently have enough content for an insight.",
    },
    stale_target: {
      label: "Stale target",
      message: "This insight may no longer match the current document structure.",
    },
    generation_failed: {
      label: "Generation failed",
      message: "Insight generation failed.",
    },
    validation_failed: {
      label: "Validation failed",
      message: "Insight response validation failed.",
    },
    evaluation_failed: {
      label: "Evaluation failed",
      message: "Insight evaluation failed.",
    },
  };

  const canGenerate = ["idle", "not_generated", "generation_failed", "validation_failed"].includes(
    insight.status,
  );
  const canRefresh = ["completed", "stale_target", "generation_failed", "validation_failed"].includes(
    insight.status,
  );

  return {
    ...statusMessages[insight.status],
    canGenerate,
    canRefresh,
    showCommands: canGenerate || canRefresh,
  };
}
