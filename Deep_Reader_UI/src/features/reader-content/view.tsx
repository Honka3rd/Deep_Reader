import MoreVertIcon from "@mui/icons-material/MoreVert";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import RefreshIcon from "@mui/icons-material/Refresh";
import PsychologyAltOutlinedIcon from "@mui/icons-material/PsychologyAltOutlined";
import QuizOutlinedIcon from "@mui/icons-material/QuizOutlined";
import TipsAndUpdatesOutlinedIcon from "@mui/icons-material/TipsAndUpdatesOutlined";
import { summaryNoteMenuItems } from "../reading-interactions/SummaryNoteMenuItems";
import {
  Box,
  Button,
  Chip,
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Stack,
  Typography,
  Tooltip,
} from "@mui/material";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { RequestStatus, SectionSelection } from "../../types/api";
import { StateView } from "../../shared/components/StateView";
import {
  InlineInsightRegion,
  type InsightViewState,
  type ReadingInteractionKind,
} from "../reading-interactions";
import {
  contentBlockInteractionAvailabilityHint,
  contentBlockInteractionEligibility,
  headingFromSelection,
  paginateMeasuredGroups,
  type ContentBlockInteractionAvailabilityHint,
  type ReaderContentBlock,
  type ReaderContentGroup,
  type ReaderContentPage,
} from "./model";

interface ReaderContentViewProps {
  onEditContentBlock?: (block: ReaderContentBlock, action: "edit" | "delete") => void;
  layoutStatus: RequestStatus;
  contentStatus: RequestStatus;
  selectedSection: SectionSelection | null;
  contentBlocks: ReaderContentBlock[];
  contentGroups: ReaderContentGroup[];
  error: string;
  onRetryContent?: () => void;
  onSelectTaskUnitInteraction?: (
    selection: SectionSelection,
    group: ReaderContentGroup,
    kind: ReadingInteractionKind,
  ) => void;
  onSelectContentBlockInteraction?: (
    selection: SectionSelection,
    block: ReaderContentBlock,
    kind: ReadingInteractionKind,
  ) => void;
  getTaskUnitInlineInsight?: (
    selection: SectionSelection,
    group: ReaderContentGroup,
  ) => InsightViewState | null;
  getContentBlockInlineInsight?: (
    selection: SectionSelection,
    block: ReaderContentBlock,
  ) => InsightViewState | null;
  onDismissInlineInsight?: (targetKey: string) => void;
  onGenerateInlineInsight?: (targetKey: string) => void;
  onRefreshInlineInsight?: (targetKey: string) => void;
}

export function ReaderContentView({
  onEditContentBlock,
  layoutStatus,
  contentStatus,
  selectedSection,
  contentBlocks,
  contentGroups,
  error,
  onRetryContent,
  onSelectTaskUnitInteraction,
  onSelectContentBlockInteraction,
  getTaskUnitInlineInsight,
  getContentBlockInlineInsight,
  onDismissInlineInsight,
  onGenerateInlineInsight,
  onRefreshInlineInsight,
}: ReaderContentViewProps) {
  const pageListRef = useRef<HTMLDivElement | null>(null);
  const measurementRef = useRef<HTMLDivElement | null>(null);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [pages, setPages] = useState<ReaderContentPage[]>([]);
  const [taskUnitInteractionMenu, setTaskUnitInteractionMenu] = useState<{
    anchorEl: HTMLElement;
    group: ReaderContentGroup;
  } | null>(null);
  const [contentBlockInteractionMenu, setContentBlockInteractionMenu] = useState<{
    anchorEl: HTMLElement;
    block: ReaderContentBlock;
  } | null>(null);
  const taskUnitInteractionMenuOpen = Boolean(taskUnitInteractionMenu);
  const contentBlockInteractionMenuOpen = Boolean(contentBlockInteractionMenu);

  useEffect(() => {
    setCurrentPageIndex(0);
  }, [selectedSection?.section.section_id, contentGroups]);

  useEffect(() => {
    if (contentStatus !== "success" || contentGroups.length === 0) {
      setPages([]);
      return;
    }

    const updatePages = () => {
      const pageList = pageListRef.current;
      const measurementContainer = measurementRef.current;
      if (!pageList || !measurementContainer) {
        return;
      }
      const groupElements = Array.from(
        measurementContainer.querySelectorAll<HTMLElement>("[data-measure-group]"),
      );
      const groupHeights = groupElements.map((element) => {
        const rect = element.getBoundingClientRect();
        return Math.ceil(rect.height);
      });
      const nextPages = paginateMeasuredGroups(
        groupHeights,
        Math.floor(pageList.getBoundingClientRect().height),
      );
      setPages(nextPages);
      setCurrentPageIndex((current) => Math.min(current, Math.max(0, nextPages.length - 1)));
    };

    updatePages();
    const resizeObserver = new ResizeObserver(updatePages);
    if (pageListRef.current) {
      resizeObserver.observe(pageListRef.current);
    }
    if (measurementRef.current) {
      resizeObserver.observe(measurementRef.current);
    }
    window.addEventListener("resize", updatePages);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updatePages);
    };
  }, [contentStatus, contentGroups]);

  const activePage = pages[currentPageIndex] || null;
  const activeGroups = useMemo(() => {
    if (!activePage) {
      return contentGroups;
    }
    return activePage.groupIndexes
      .map((groupIndex) => contentGroups[groupIndex])
      .filter((group): group is ReaderContentGroup => Boolean(group));
  }, [activePage, contentGroups]);

  const pageCount = pages.length || (contentStatus === "success" ? 1 : 0);
  const displayPageIndex = Math.min(currentPageIndex, Math.max(0, pageCount - 1));
  const canGoPrevious = displayPageIndex > 0;
  const canGoNext = displayPageIndex + 1 < pageCount;

  function openTaskUnitInteractionMenu(
    event: MouseEvent<HTMLButtonElement>,
    group: ReaderContentGroup,
  ) {
    setTaskUnitInteractionMenu({ anchorEl: event.currentTarget, group });
  }

  function closeTaskUnitInteractionMenu() {
    setTaskUnitInteractionMenu(null);
  }

  function openContentBlockInteractionMenu(
    event: MouseEvent<HTMLButtonElement>,
    block: ReaderContentBlock,
  ) {
    setContentBlockInteractionMenu({ anchorEl: event.currentTarget, block });
  }

  function closeContentBlockInteractionMenu() {
    setContentBlockInteractionMenu(null);
  }

  function selectTaskUnitInteraction(kind: ReadingInteractionKind) {
    const group = taskUnitInteractionMenu?.group;
    closeTaskUnitInteractionMenu();
    if (selectedSection && group) {
      onSelectTaskUnitInteraction?.(selectedSection, group, kind);
    }
  }

  function selectContentBlockInteraction(kind: ReadingInteractionKind) {
    const block = contentBlockInteractionMenu?.block;
    closeContentBlockInteractionMenu();
    if (selectedSection && block) {
      onSelectContentBlockInteraction?.(selectedSection, block, kind);
    }
  }

  function contentBlockMenuHint(
    kind: ReadingInteractionKind,
  ): ContentBlockInteractionAvailabilityHint | null {
    const block = contentBlockInteractionMenu?.block;
    return block ? contentBlockInteractionAvailabilityHint(block, kind) : null;
  }

  function formatContentBlockMenuHint(
    hint: ContentBlockInteractionAvailabilityHint | null,
  ): string | undefined {
    if (!hint) {
      return undefined;
    }
    return hint.reason ? `${hint.label}: ${hint.reason}` : hint.label;
  }

  if (layoutStatus === "initial") {
    return (
      <StateView
        className="reader-content-initial-state"
        title="Ready"
        detail="Enter an existing document name to load reading units."
      />
    );
  }

  if (layoutStatus === "loading") {
    return <StateView className="reader-content-loading-state" title="Loading document" loading />;
  }

  if (layoutStatus === "error") {
    return <StateView className="reader-content-unavailable-state" title="No content loaded" />;
  }

  if (!selectedSection) {
    return (
      <StateView
        className="reader-content-no-selection-state"
        title="No section selected"
        detail="Choose a section from the hierarchy."
      />
    );
  }

  return (
    <Box className="reader-content">
      <Box component="header" className="content-header reader-content-header">
        <Typography className="eyebrow reader-content-context" color="text.secondary">
          {[selectedSection.section.container_title, selectedSection.chapter.title]
            .filter(Boolean)
            .join(" / ")}
        </Typography>
        <Typography className="reader-content-title" component="h1" variant="h1">
          {headingFromSelection(selectedSection)}
        </Typography>
      </Box>

      {contentStatus === "loading" ? (
        <StateView
          className="reader-content-blocks-loading-state"
          title="Loading content"
          loading
        />
      ) : null}
      {contentStatus === "error" ? (
        <Box>
          <StateView
            className="reader-content-blocks-error-state"
            title="Section content request failed"
            detail={error}
            severity="error"
          />
          <Tooltip title="Retry content">
            <IconButton aria-label="Retry content" onClick={onRetryContent} disabled={!onRetryContent}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      ) : null}
      {contentStatus === "empty" ? (
        <StateView className="reader-content-blocks-empty-state" title="No content blocks" />
      ) : null}

      {contentStatus === "success" ? (
        <>
          <Box className="reader-content-pagination-bar">
            <Typography color="text.secondary" variant="body2">
              Page {displayPageIndex + 1} / {pageCount}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button
                className="reader-content-page-previous"
                size="small"
                variant="outlined"
                disabled={!canGoPrevious}
                onClick={() => setCurrentPageIndex((current) => Math.max(0, current - 1))}
              >
                Previous
              </Button>
              <Button
                className="reader-content-page-next"
                size="small"
                variant="outlined"
                disabled={!canGoNext}
                onClick={() =>
                  setCurrentPageIndex((current) => Math.min(pageCount - 1, current + 1))
                }
              >
                Next
              </Button>
            </Stack>
          </Box>
          <Stack
            component="article"
            ref={pageListRef}
            spacing={2.25}
            className={[
              "content-blocks",
              "reader-content-block-list",
              activePage?.oversized ? "reader-content-block-list-oversized" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            {activeGroups.length > 0
              ? activeGroups.map((group) => (
                  <ReaderContentGroupView
                    group={group}
                    insight={
                      selectedSection ? getTaskUnitInlineInsight?.(selectedSection, group) : null
                    }
                    selectedSection={selectedSection}
                    key={group.taskUnitId}
                    getContentBlockInlineInsight={getContentBlockInlineInsight}
                    onDismissInlineInsight={onDismissInlineInsight}
                    onGenerateInlineInsight={onGenerateInlineInsight}
                    onOpenContentBlockInteractionMenu={openContentBlockInteractionMenu}
                    onOpenInteractionMenu={openTaskUnitInteractionMenu}
                    onRefreshInlineInsight={onRefreshInlineInsight}
                    showInteraction
                  />
                ))
              : contentBlocks.map((block, index) => (
                  <ContentBlockView
                    block={block}
                    insight={
                      selectedSection
                        ? getContentBlockInlineInsight?.(selectedSection, block)
                        : null
                    }
                    index={index}
                    key={`${block.block_id}:${index}`}
                    onDismissInlineInsight={onDismissInlineInsight}
                    onGenerateInlineInsight={onGenerateInlineInsight}
                    onOpenInteractionMenu={openContentBlockInteractionMenu}
                    onRefreshInlineInsight={onRefreshInlineInsight}
                    showInteraction
                  />
                ))}
          </Stack>
          <Menu
            className="reader-task-unit-interaction-menu"
            id="reader-task-unit-interaction-menu"
            anchorEl={taskUnitInteractionMenu?.anchorEl || null}
            open={taskUnitInteractionMenuOpen}
            onClose={closeTaskUnitInteractionMenu}
          >
            {summaryNoteMenuItems(selectTaskUnitInteraction)}
            <MenuItem
              className="reader-task-unit-interaction-option reader-task-unit-interaction-option-insight"
              onClick={() => selectTaskUnitInteraction("insight")}
            >
              <ListItemIcon className="reader-task-unit-interaction-option-icon">
                <TipsAndUpdatesOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText className="reader-task-unit-interaction-option-label">
                Insights
              </ListItemText>
            </MenuItem>
            <MenuItem
              className="reader-task-unit-interaction-option reader-task-unit-interaction-option-quiz"
              onClick={() => selectTaskUnitInteraction("quiz")}
            >
              <ListItemIcon className="reader-task-unit-interaction-option-icon">
                <QuizOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText className="reader-task-unit-interaction-option-label">
                Quiz
              </ListItemText>
            </MenuItem>
            <MenuItem
              className="reader-task-unit-interaction-option reader-task-unit-interaction-option-critical-thinking"
              onClick={() => selectTaskUnitInteraction("critical_thinking")}
            >
              <ListItemIcon className="reader-task-unit-interaction-option-icon">
                <PsychologyAltOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText className="reader-task-unit-interaction-option-label">
                Critical thinking
              </ListItemText>
            </MenuItem>
          </Menu>
          <Menu
            className="reader-content-block-interaction-menu"
            id="reader-content-block-interaction-menu"
            anchorEl={contentBlockInteractionMenu?.anchorEl || null}
            open={contentBlockInteractionMenuOpen}
            onClose={closeContentBlockInteractionMenu}
          >
            {summaryNoteMenuItems(selectContentBlockInteraction)}
            {onEditContentBlock && ["edit", "delete"].map(action => <MenuItem key={action} onClick={() => {
              const block = contentBlockInteractionMenu?.block;
              closeContentBlockInteractionMenu();
              if (block) onEditContentBlock(block, action as "edit" | "delete");
            }}><ListItemIcon>{action === "edit" ? <EditOutlinedIcon fontSize="small" /> : <DeleteOutlineIcon fontSize="small" />}</ListItemIcon>
              <ListItemText primary={action === "edit" ? "Edit" : "Delete"} /></MenuItem>)}
            <MenuItem
              className="reader-content-block-interaction-option reader-content-block-interaction-option-insight"
              onClick={() => selectContentBlockInteraction("insight")}
            >
              <ListItemIcon className="reader-content-block-interaction-option-icon">
                <TipsAndUpdatesOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText
                className="reader-content-block-interaction-option-label"
                primary="Insights"
                secondary={formatContentBlockMenuHint(contentBlockMenuHint("insight"))}
              />
            </MenuItem>
            <MenuItem
              className="reader-content-block-interaction-option reader-content-block-interaction-option-quiz"
              onClick={() => selectContentBlockInteraction("quiz")}
            >
              <ListItemIcon className="reader-content-block-interaction-option-icon">
                <QuizOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText
                className="reader-content-block-interaction-option-label"
                primary="Quiz"
                secondary={formatContentBlockMenuHint(contentBlockMenuHint("quiz"))}
              />
            </MenuItem>
            <MenuItem
              className="reader-content-block-interaction-option reader-content-block-interaction-option-critical-thinking"
              onClick={() => selectContentBlockInteraction("critical_thinking")}
            >
              <ListItemIcon className="reader-content-block-interaction-option-icon">
                <PsychologyAltOutlinedIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText
                className="reader-content-block-interaction-option-label"
                primary="Critical thinking"
                secondary={formatContentBlockMenuHint(
                  contentBlockMenuHint("critical_thinking"),
                )}
              />
            </MenuItem>
          </Menu>
          <Box
            aria-hidden="true"
            className="reader-content-pagination-measure"
            ref={measurementRef}
          >
            {contentGroups.map((group) => (
              <Stack
                className="reader-content-measure-group"
                data-measure-group
                key={group.taskUnitId}
                spacing={2.25}
              >
                <ReaderContentGroupView group={group} showInteraction={false} />
              </Stack>
            ))}
          </Box>
        </>
      ) : null}
    </Box>
  );
}

function ReaderContentGroupView({
  group,
  insight,
  selectedSection,
  showInteraction,
  getContentBlockInlineInsight,
  onDismissInlineInsight,
  onGenerateInlineInsight,
  onOpenContentBlockInteractionMenu,
  onOpenInteractionMenu,
  onRefreshInlineInsight,
}: {
  group: ReaderContentGroup;
  insight?: InsightViewState | null;
  selectedSection?: SectionSelection | null;
  showInteraction: boolean;
  getContentBlockInlineInsight?: (
    selection: SectionSelection,
    block: ReaderContentBlock,
  ) => InsightViewState | null;
  onDismissInlineInsight?: (targetKey: string) => void;
  onGenerateInlineInsight?: (targetKey: string) => void;
  onOpenContentBlockInteractionMenu?: (
    event: MouseEvent<HTMLButtonElement>,
    block: ReaderContentBlock,
  ) => void;
  onOpenInteractionMenu?: (
    event: MouseEvent<HTMLButtonElement>,
    group: ReaderContentGroup,
  ) => void;
  onRefreshInlineInsight?: (targetKey: string) => void;
}) {
  const title = group.title?.trim() || group.taskUnitId;

  return (
    <Box className="reader-content-task-unit-group">
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        className="reader-content-task-unit-header"
      >
        <Box className="reader-content-task-unit-heading">
          <Typography className="reader-content-task-unit-title" component="h2" variant="h2">
            {title}
          </Typography>
          <Typography
            className="reader-content-task-unit-id"
            variant="caption"
            color="text.secondary"
          >
            {group.taskUnitId}
          </Typography>
        </Box>
        {showInteraction ? (
          <IconButton
            className="reader-task-unit-interaction-trigger"
            aria-label={`Task-unit reading interactions for ${title}`}
            size="small"
            onClick={(event) => onOpenInteractionMenu?.(event, group)}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>
        ) : null}
      </Stack>
      {insight ? (
        <InlineInsightRegion
          className="reader-task-unit-inline-insight"
          insight={insight}
          onDismiss={onDismissInlineInsight}
          onGenerate={onGenerateInlineInsight}
          onRefresh={onRefreshInlineInsight}
        />
      ) : null}
      <Stack spacing={2.25} className="reader-content-task-unit-blocks">
        {group.blocks.length > 0 ? (
          group.blocks.map((block, index) => (
            <ContentBlockView
              block={block}
              insight={
                selectedSection
                  ? getContentBlockInlineInsight?.(selectedSection, block)
                  : null
              }
              index={index}
              key={`${group.taskUnitId}:${block.block_id}:${index}`}
              onDismissInlineInsight={onDismissInlineInsight}
              onGenerateInlineInsight={onGenerateInlineInsight}
              onOpenInteractionMenu={onOpenContentBlockInteractionMenu}
              onRefreshInlineInsight={onRefreshInlineInsight}
              showInteraction={showInteraction}
            />
          ))
        ) : (
          <Typography
            className="reader-content-task-unit-empty"
            color="text.secondary"
            variant="body2"
          >
            No renderable content blocks for this task unit.
          </Typography>
        )}
      </Stack>
    </Box>
  );
}

function ContentBlockView({
  block,
  insight,
  index,
  showInteraction,
  onDismissInlineInsight,
  onGenerateInlineInsight,
  onOpenInteractionMenu,
  onRefreshInlineInsight,
}: {
  block: ReaderContentBlock;
  insight?: InsightViewState | null;
  index: number;
  showInteraction: boolean;
  onDismissInlineInsight?: (targetKey: string) => void;
  onGenerateInlineInsight?: (targetKey: string) => void;
  onOpenInteractionMenu?: (
    event: MouseEvent<HTMLButtonElement>,
    block: ReaderContentBlock,
  ) => void;
  onRefreshInlineInsight?: (targetKey: string) => void;
}) {
  const interactionEligibility = contentBlockInteractionEligibility(block);
  const showBlockInteraction = showInteraction && interactionEligibility.eligible;
  const blockLabel = block.block_id.trim() || `content block ${index + 1}`;

  return (
    <Box
      component="section"
      className={[
        "content-block",
        "reader-content-block",
        interactionEligibility.eligible
          ? "reader-content-block-interaction-eligible"
          : "reader-content-block-render-only",
      ]
        .filter(Boolean)
        .join(" ")}
      data-content-block-interaction-eligible={
        interactionEligibility.eligible ? "true" : "false"
      }
      data-content-block-interaction-reason={interactionEligibility.reason}
      key={`${block.block_id}:${index}`}
    >
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        className="block-meta reader-content-block-meta"
      >
        <Chip
          className="reader-content-block-type"
          size="small"
          label={block.block_type || "content"}
        />
        <Typography
          className="reader-content-block-id"
          variant="caption"
          color="text.secondary"
        >
          {block.block_id}
        </Typography>
        {showBlockInteraction ? (
          <IconButton
            className="reader-content-block-interaction-trigger"
            aria-label={`Content-block reading interactions for ${blockLabel}`}
            size="small"
            onClick={(event) => onOpenInteractionMenu?.(event, block)}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>
        ) : null}
      </Stack>
      <Typography className="reader-content-block-body" variant="body1">
        {block.content}
      </Typography>
      {insight ? (
        <InlineInsightRegion
          className="reader-content-block-inline-insight"
          insight={insight}
          onDismiss={onDismissInlineInsight}
          onGenerate={onGenerateInlineInsight}
          onRefresh={onRefreshInlineInsight}
        />
      ) : null}
    </Box>
  );
}
