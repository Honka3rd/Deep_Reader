import MoreVertIcon from "@mui/icons-material/MoreVert";
import PsychologyAltOutlinedIcon from "@mui/icons-material/PsychologyAltOutlined";
import QuizOutlinedIcon from "@mui/icons-material/QuizOutlined";
import TipsAndUpdatesOutlinedIcon from "@mui/icons-material/TipsAndUpdatesOutlined";
import {
  Box,
  Button,
  ButtonBase,
  Chip,
  Divider,
  IconButton,
  List,
  ListItemIcon,
  ListItemText,
  ListItem,
  Menu,
  MenuItem,
  Stack,
  Typography,
} from "@mui/material";
import { useState, type MouseEvent } from "react";
import type {
  ChapterLayout,
  DocumentTaskLayout,
  RequestStatus,
  SectionLayout,
  SectionSelection,
} from "../../types/api";
import { StateView } from "../../shared/components/StateView";
import {
  InlineInsightRegion,
  type InsightViewState,
  type ReadingInteractionKind,
} from "../reading-interactions";
import {
  buildSectionDisplay,
  countTaskUnits,
  labelOrId,
} from "./model";

interface HierarchyNavigationViewProps {
  layout: DocumentTaskLayout | null;
  selectedSectionId: string | null;
  status: RequestStatus;
  error: string;
  canEditToc?: boolean;
  onSelectSection: (selection: SectionSelection) => void;
  onEditToc?: () => void;
  onSelectDocumentInteraction?: (kind: ReadingInteractionKind) => void;
  onSelectChapterInteraction?: (
    chapter: ChapterLayout,
    kind: ReadingInteractionKind,
  ) => void;
  onSelectSectionInteraction?: (
    chapter: ChapterLayout,
    section: SectionLayout,
    kind: ReadingInteractionKind,
  ) => void;
  documentInlineInsight?: InsightViewState | null;
  getChapterInlineInsight?: (chapter: ChapterLayout) => InsightViewState | null;
  getSectionInlineInsight?: (
    chapter: ChapterLayout,
    section: SectionLayout,
  ) => InsightViewState | null;
  onDismissInlineInsight?: (targetKey: string) => void;
  onGenerateInlineInsight?: (targetKey: string) => void;
  onRefreshInlineInsight?: (targetKey: string) => void;
}

function ChapterInteractionButton({
  chapter,
  chapterTitle,
  onOpen,
}: {
  chapter: ChapterLayout;
  chapterTitle: string;
  onOpen: (event: MouseEvent<HTMLButtonElement>, chapter: ChapterLayout) => void;
}) {
  return (
    <IconButton
      className="hierarchy-chapter-interaction-trigger"
      aria-label={`Chapter reading interactions for ${chapterTitle}`}
      size="small"
      onClick={(event) => onOpen(event, chapter)}
    >
      <MoreVertIcon fontSize="small" />
    </IconButton>
  );
}

function SectionButton({
  chapter,
  section,
  selected,
  displayTitle,
  displaySubtitle,
  onSelectSection,
}: {
  chapter: ChapterLayout;
  section: SectionLayout;
  selected: boolean;
  displayTitle?: string;
  displaySubtitle?: string | null;
  onSelectSection: (selection: SectionSelection) => void;
}) {
  const taskUnits = section.task_units || [];
  const resolvedTitle = displayTitle || labelOrId(section.title, section.section_id);
  const buttonClassName = [
    "section-button",
    "hierarchy-section-button",
    selected ? "selected hierarchy-section-button-selected" : "",
    taskUnits.length === 0 ? "hierarchy-section-button-empty" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <ButtonBase
      className={buttonClassName}
      aria-selected={selected}
      disabled={taskUnits.length === 0}
      onClick={() => onSelectSection({ chapter, section, taskUnits })}
    >
      <Box className="section-button-copy hierarchy-section-label">
        <Typography className="hierarchy-section-title" component="span" variant="body1">
          {resolvedTitle}
        </Typography>
        {displaySubtitle || section.container_title ? (
          <Typography
            className="hierarchy-section-subtitle"
            component="span"
            variant="caption"
            color="text.secondary"
          >
            {displaySubtitle || section.container_title}
          </Typography>
        ) : null}
      </Box>
      <Chip
        className="hierarchy-section-unit-count"
        size="small"
        label={`${taskUnits.length} units`}
      />
    </ButtonBase>
  );
}

function SectionInteractionButton({
  chapter,
  section,
  sectionTitle,
  onOpen,
}: {
  chapter: ChapterLayout;
  section: SectionLayout;
  sectionTitle: string;
  onOpen: (
    event: MouseEvent<HTMLButtonElement>,
    chapter: ChapterLayout,
    section: SectionLayout,
  ) => void;
}) {
  return (
    <IconButton
      className="hierarchy-section-interaction-trigger"
      aria-label={`Section reading interactions for ${sectionTitle}`}
      size="small"
      onClick={(event) => onOpen(event, chapter, section)}
    >
      <MoreVertIcon fontSize="small" />
    </IconButton>
  );
}

export function HierarchyNavigationView({
  layout,
  selectedSectionId,
  status,
  error,
  canEditToc = false,
  onSelectSection,
  onEditToc,
  onSelectDocumentInteraction,
  onSelectChapterInteraction,
  onSelectSectionInteraction,
  documentInlineInsight,
  getChapterInlineInsight,
  getSectionInlineInsight,
  onDismissInlineInsight,
  onGenerateInlineInsight,
  onRefreshInlineInsight,
}: HierarchyNavigationViewProps) {
  const [documentInteractionMenuAnchor, setDocumentInteractionMenuAnchor] =
    useState<HTMLElement | null>(null);
  const [chapterInteractionMenu, setChapterInteractionMenu] = useState<{
    anchorEl: HTMLElement;
    chapter: ChapterLayout;
  } | null>(null);
  const [sectionInteractionMenu, setSectionInteractionMenu] = useState<{
    anchorEl: HTMLElement;
    chapter: ChapterLayout;
    section: SectionLayout;
  } | null>(null);
  const documentInteractionMenuOpen = Boolean(documentInteractionMenuAnchor);
  const chapterInteractionMenuOpen = Boolean(chapterInteractionMenu);
  const sectionInteractionMenuOpen = Boolean(sectionInteractionMenu);

  function openDocumentInteractionMenu(event: MouseEvent<HTMLButtonElement>) {
    setDocumentInteractionMenuAnchor(event.currentTarget);
  }

  function closeDocumentInteractionMenu() {
    setDocumentInteractionMenuAnchor(null);
  }

  function selectDocumentInteraction(kind: ReadingInteractionKind) {
    closeDocumentInteractionMenu();
    onSelectDocumentInteraction?.(kind);
  }

  function openChapterInteractionMenu(
    event: MouseEvent<HTMLButtonElement>,
    chapter: ChapterLayout,
  ) {
    event.stopPropagation();
    setChapterInteractionMenu({ anchorEl: event.currentTarget, chapter });
  }

  function closeChapterInteractionMenu() {
    setChapterInteractionMenu(null);
  }

  function selectChapterInteraction(kind: ReadingInteractionKind) {
    const chapter = chapterInteractionMenu?.chapter;
    closeChapterInteractionMenu();
    if (chapter) {
      onSelectChapterInteraction?.(chapter, kind);
    }
  }

  function openSectionInteractionMenu(
    event: MouseEvent<HTMLButtonElement>,
    chapter: ChapterLayout,
    section: SectionLayout,
  ) {
    event.stopPropagation();
    setSectionInteractionMenu({ anchorEl: event.currentTarget, chapter, section });
  }

  function closeSectionInteractionMenu() {
    setSectionInteractionMenu(null);
  }

  function selectSectionInteraction(kind: ReadingInteractionKind) {
    const target = sectionInteractionMenu;
    closeSectionInteractionMenu();
    if (target) {
      onSelectSectionInteraction?.(target.chapter, target.section, kind);
    }
  }

  if (status === "initial") {
    return <StateView className="hierarchy-empty-state" title="No document loaded" />;
  }

  if (status === "loading") {
    return <StateView className="hierarchy-loading-state" title="Loading hierarchy" loading />;
  }

  if (status === "error") {
    return (
      <StateView
        className="hierarchy-error-state"
        title="Task layout request failed"
        detail={error}
        severity="error"
      />
    );
  }

  if (!layout?.chapters?.length || countTaskUnits(layout) === 0) {
    return (
      <StateView
        className="hierarchy-no-task-units-state"
        title="No task units"
        detail="The backend returned no navigable task units for this document."
      />
    );
  }

  return (
    <Box className="hierarchy-navigation">
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        className="nav-title-row hierarchy-navigation-header"
      >
        <Typography className="hierarchy-document-title" component="h1" variant="h2">
          {layout.title || "Document"}
        </Typography>
        <Chip
          className="hierarchy-document-unit-count"
          size="small"
          label={`${countTaskUnits(layout)} units`}
        />
        <Button
          className="hierarchy-edit-toc-trigger"
          size="small"
          variant="outlined"
          disabled={!canEditToc}
          onClick={onEditToc}
        >
          Edit TOC
        </Button>
        <IconButton
          className="hierarchy-document-interaction-trigger"
          id="document-interaction-menu-button"
          aria-label="Document reading interactions"
          aria-controls={documentInteractionMenuOpen ? "document-interaction-menu" : undefined}
          aria-haspopup="menu"
          aria-expanded={documentInteractionMenuOpen ? "true" : undefined}
          size="small"
          onClick={openDocumentInteractionMenu}
        >
          <MoreVertIcon fontSize="small" />
        </IconButton>
        <Menu
          className="hierarchy-document-interaction-menu"
          id="document-interaction-menu"
          anchorEl={documentInteractionMenuAnchor}
          open={documentInteractionMenuOpen}
          onClose={closeDocumentInteractionMenu}
          MenuListProps={{ "aria-labelledby": "document-interaction-menu-button" }}
        >
          <MenuItem
            className="hierarchy-document-interaction-option hierarchy-document-interaction-option-insight"
            onClick={() => selectDocumentInteraction("insight")}
          >
            <ListItemIcon className="hierarchy-document-interaction-option-icon">
              <TipsAndUpdatesOutlinedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText className="hierarchy-document-interaction-option-label">
              Insights
            </ListItemText>
          </MenuItem>
          <MenuItem
            className="hierarchy-document-interaction-option hierarchy-document-interaction-option-quiz"
            onClick={() => selectDocumentInteraction("quiz")}
          >
            <ListItemIcon className="hierarchy-document-interaction-option-icon">
              <QuizOutlinedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText className="hierarchy-document-interaction-option-label">
              Quiz
            </ListItemText>
          </MenuItem>
          <MenuItem
            className="hierarchy-document-interaction-option hierarchy-document-interaction-option-critical-thinking"
            onClick={() => selectDocumentInteraction("critical_thinking")}
          >
            <ListItemIcon className="hierarchy-document-interaction-option-icon">
              <PsychologyAltOutlinedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText className="hierarchy-document-interaction-option-label">
              Critical thinking
            </ListItemText>
          </MenuItem>
        </Menu>
      </Stack>
      {documentInlineInsight ? (
        <InlineInsightRegion
          className="hierarchy-document-inline-insight"
          insight={documentInlineInsight}
          onDismiss={onDismissInlineInsight}
          onGenerate={onGenerateInlineInsight}
          onRefresh={onRefreshInlineInsight}
        />
      ) : null}
      <List component="ol" className="chapter-list hierarchy-chapter-list" disablePadding>
        {layout.chapters.map((chapter) => {
          const sections = chapter.sections || [];
          const singleSection = sections.length === 1 ? sections[0] : null;
          const chapterTitle = labelOrId(chapter.title, chapter.chapter_id);
          const mergedSectionDisplay = singleSection
            ? buildSectionDisplay(chapter, singleSection)
            : null;
          const chapterInlineInsight = getChapterInlineInsight?.(chapter) || null;

          return (
            <ListItem
              component="li"
              className={
                singleSection
                  ? "chapter-item hierarchy-chapter-item hierarchy-chapter-item-merged merged"
                  : "chapter-item hierarchy-chapter-item"
              }
              key={chapter.chapter_id}
            >
              {singleSection ? (
                <Box className="hierarchy-chapter-merged-row">
                  <SectionButton
                    chapter={chapter}
                    section={singleSection}
                    selected={selectedSectionId === singleSection.section_id}
                    displayTitle={mergedSectionDisplay?.title}
                    displaySubtitle={mergedSectionDisplay?.subtitle}
                    onSelectSection={onSelectSection}
                  />
                  <ChapterInteractionButton
                    chapter={chapter}
                    chapterTitle={chapterTitle}
                    onOpen={openChapterInteractionMenu}
                  />
                </Box>
              ) : (
                <>
                  <Box className="hierarchy-chapter-header-row">
                    <Typography className="hierarchy-chapter-title" component="h2" variant="h2">
                      {chapterTitle}
                    </Typography>
                    <ChapterInteractionButton
                      chapter={chapter}
                      chapterTitle={chapterTitle}
                      onOpen={openChapterInteractionMenu}
                    />
                  </Box>
                  {chapterInlineInsight ? (
                    <InlineInsightRegion
                      className="hierarchy-chapter-inline-insight"
                      insight={chapterInlineInsight}
                      onDismiss={onDismissInlineInsight}
                      onGenerate={onGenerateInlineInsight}
                      onRefresh={onRefreshInlineInsight}
                    />
                  ) : null}
                  <List
                    component="ol"
                    className="section-list hierarchy-section-list"
                    disablePadding
                  >
                    {sections.map((section) => {
                      const sectionTitle = labelOrId(section.title, section.section_id);
                      const sectionInlineInsight =
                        getSectionInlineInsight?.(chapter, section) || null;

                      return (
                        <ListItem
                          component="li"
                          className="section-item hierarchy-section-item"
                          key={section.section_id}
                        >
                          <Box className="hierarchy-section-row">
                            <SectionButton
                              chapter={chapter}
                              section={section}
                              selected={selectedSectionId === section.section_id}
                              onSelectSection={onSelectSection}
                            />
                            <SectionInteractionButton
                              chapter={chapter}
                              section={section}
                              sectionTitle={sectionTitle}
                              onOpen={openSectionInteractionMenu}
                            />
                          </Box>
                          {sectionInlineInsight ? (
                            <InlineInsightRegion
                              className="hierarchy-section-inline-insight"
                              insight={sectionInlineInsight}
                              onDismiss={onDismissInlineInsight}
                              onGenerate={onGenerateInlineInsight}
                              onRefresh={onRefreshInlineInsight}
                            />
                          ) : null}
                        </ListItem>
                      );
                    })}
                  </List>
                </>
              )}
              {singleSection && chapterInlineInsight ? (
                <InlineInsightRegion
                  className="hierarchy-chapter-inline-insight"
                  insight={chapterInlineInsight}
                  onDismiss={onDismissInlineInsight}
                  onGenerate={onGenerateInlineInsight}
                  onRefresh={onRefreshInlineInsight}
                />
              ) : null}
              <Divider />
            </ListItem>
          );
        })}
      </List>
      <Menu
        className="hierarchy-chapter-interaction-menu"
        id="chapter-interaction-menu"
        anchorEl={chapterInteractionMenu?.anchorEl || null}
        open={chapterInteractionMenuOpen}
        onClose={closeChapterInteractionMenu}
      >
        <MenuItem
          className="hierarchy-chapter-interaction-option hierarchy-chapter-interaction-option-insight"
          onClick={() => selectChapterInteraction("insight")}
        >
          <ListItemIcon className="hierarchy-chapter-interaction-option-icon">
            <TipsAndUpdatesOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-chapter-interaction-option-label">
            Insights
          </ListItemText>
        </MenuItem>
        <MenuItem
          className="hierarchy-chapter-interaction-option hierarchy-chapter-interaction-option-quiz"
          onClick={() => selectChapterInteraction("quiz")}
        >
          <ListItemIcon className="hierarchy-chapter-interaction-option-icon">
            <QuizOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-chapter-interaction-option-label">
            Quiz
          </ListItemText>
        </MenuItem>
        <MenuItem
          className="hierarchy-chapter-interaction-option hierarchy-chapter-interaction-option-critical-thinking"
          onClick={() => selectChapterInteraction("critical_thinking")}
        >
          <ListItemIcon className="hierarchy-chapter-interaction-option-icon">
            <PsychologyAltOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-chapter-interaction-option-label">
            Critical thinking
          </ListItemText>
        </MenuItem>
      </Menu>
      <Menu
        className="hierarchy-section-interaction-menu"
        id="section-interaction-menu"
        anchorEl={sectionInteractionMenu?.anchorEl || null}
        open={sectionInteractionMenuOpen}
        onClose={closeSectionInteractionMenu}
      >
        <MenuItem
          className="hierarchy-section-interaction-option hierarchy-section-interaction-option-insight"
          onClick={() => selectSectionInteraction("insight")}
        >
          <ListItemIcon className="hierarchy-section-interaction-option-icon">
            <TipsAndUpdatesOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-section-interaction-option-label">
            Insights
          </ListItemText>
        </MenuItem>
        <MenuItem
          className="hierarchy-section-interaction-option hierarchy-section-interaction-option-quiz"
          onClick={() => selectSectionInteraction("quiz")}
        >
          <ListItemIcon className="hierarchy-section-interaction-option-icon">
            <QuizOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-section-interaction-option-label">
            Quiz
          </ListItemText>
        </MenuItem>
        <MenuItem
          className="hierarchy-section-interaction-option hierarchy-section-interaction-option-critical-thinking"
          onClick={() => selectSectionInteraction("critical_thinking")}
        >
          <ListItemIcon className="hierarchy-section-interaction-option-icon">
            <PsychologyAltOutlinedIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText className="hierarchy-section-interaction-option-label">
            Critical thinking
          </ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
}
