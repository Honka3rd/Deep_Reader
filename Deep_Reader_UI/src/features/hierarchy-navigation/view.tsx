import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { NavigationTreeItem } from "./NavigationTreeItem";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import PsychologyAltOutlinedIcon from "@mui/icons-material/PsychologyAltOutlined";
import QuizOutlinedIcon from "@mui/icons-material/QuizOutlined";
import TipsAndUpdatesOutlinedIcon from "@mui/icons-material/TipsAndUpdatesOutlined";
import { summaryNoteMenuItems } from "../reading-interactions/SummaryNoteMenuItems";
import {
  Box,
  Alert,
  Button,
  Chip,
  IconButton,
  ListItemIcon,
  ListItemText,
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

function SectionLabel({
  section, displayTitle, displaySubtitle,
}: {
  section: SectionLayout;
  displayTitle?: string;
  displaySubtitle?: string | null;
}) {
  return (
    <Box className="section-button-copy hierarchy-section-label">
      <Typography className="hierarchy-section-title" component="span" variant="body1">
        {displayTitle || labelOrId(section.title, section.section_id)}
      </Typography>
      {section.title_origin === "generated" && <Typography component="span" variant="caption" color="text.secondary">
        Generated heading
      </Typography>}
      {displaySubtitle || section.container_title ? (
        <Typography className="hierarchy-section-subtitle" component="span" variant="caption" color="text.secondary">
          {displaySubtitle || section.container_title}
        </Typography>
      ) : null}
      <Typography className="hierarchy-section-unit-count" component="span" variant="caption" color="text.secondary">
        {(section.task_units || []).length} units
      </Typography>
    </Box>
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

  function selectTreeSection(itemId: string) {
    for (const chapter of layout?.chapters || []) {
      const section = chapter.sections?.find(item => `section:${item.section_id}` === itemId);
      if (section?.task_units?.length) {
        onSelectSection({ chapter, section, taskUnits: section.task_units });
        return;
      }
    }
  }

  return (
    <Box className="hierarchy-navigation">
      <Stack
        direction="row"
        spacing={1}
        useFlexGap
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
          {summaryNoteMenuItems(selectDocumentInteraction)}
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
      {layout.parse_provenance?.fallback_used && <Alert severity="warning">
        Requested {layout.parse_provenance.requested_parser_mode}; using {layout.parse_provenance.effective_parser_mode}.
        {layout.parse_provenance.fallback_reason ? ` ${layout.parse_provenance.fallback_reason}` : ""}
      </Alert>}
      {layout.parse_provenance?.declaration_unconfirmed && <Alert severity="warning">
        The declared printed table of contents could not be confirmed. The accepted structure remains readable.
      </Alert>}
      {layout.chapters.some(chapter => chapter.metadata?.title_origin === "generated" ||
        chapter.sections?.some(section => section.title_origin === "generated")) && <Typography variant="caption">
        Generated reading outline, not an original table of contents.
      </Typography>}
      {layout.parse_provenance?.implicit_single_body && <Typography variant="caption">
        Main Body · No validated heading boundaries
      </Typography>}
      {documentInlineInsight ? (
        <InlineInsightRegion
          className="hierarchy-document-inline-insight"
          insight={documentInlineInsight}
          onDismiss={onDismissInlineInsight}
          onGenerate={onGenerateInlineInsight}
          onRefresh={onRefreshInlineInsight}
        />
      ) : null}
      <SimpleTreeView
        key={layout.document_id}
        className="chapter-list hierarchy-chapter-list"
        aria-label="Chapters and sections"
        itemChildrenIndentation={16}
        selectedItems={selectedSectionId ? `section:${selectedSectionId}` : null}
        defaultExpandedItems={layout.chapters.filter(chapter =>
          chapter.sections?.some(section => section.section_id === selectedSectionId),
        ).map(chapter => `chapter:${chapter.chapter_id}`)}
        onItemClick={(_event, itemId) => selectTreeSection(itemId)}
        onItemSelectionToggle={(event, itemId, selected) => {
          // Mouse activation also supports re-reading the already-selected section.
          // Keyboard selection uses this callback; avoid dispatching a click twice.
          if (selected && event?.type !== "click") selectTreeSection(itemId);
        }}
      >
        {layout.chapters.map((chapter) => {
          const sections = chapter.sections || [];
          const singleSection = sections.length === 1 ? sections[0] : null;
          const chapterTitle = labelOrId(chapter.title, chapter.chapter_id);
          const display = singleSection ? buildSectionDisplay(chapter, singleSection) : null;
          const chapterInsight = getChapterInlineInsight?.(chapter);
          return (
            <NavigationTreeItem
              key={chapter.chapter_id}
              itemId={singleSection ? `section:${singleSection.section_id}` : `chapter:${chapter.chapter_id}`}
              className={`chapter-item hierarchy-chapter-item${singleSection ? " hierarchy-chapter-item-merged merged" : ""}`}
              contentClassName={singleSection ? "hierarchy-chapter-merged-row hierarchy-section-button" : "hierarchy-chapter-header-row"}
              label={display?.title || chapterTitle}
              disabled={Boolean(singleSection && !singleSection.task_units?.length)}
              copy={singleSection ? <SectionLabel section={singleSection} displayTitle={display?.title} displaySubtitle={display?.subtitle} />
                : <Typography className="hierarchy-chapter-title" component="span" variant="body1">{chapterTitle}</Typography>}
              action={<ChapterInteractionButton chapter={chapter} chapterTitle={chapterTitle} onOpen={openChapterInteractionMenu} />}
              insight={chapterInsight ? <InlineInsightRegion className="hierarchy-chapter-inline-insight"
                insight={chapterInsight} onDismiss={onDismissInlineInsight}
                onGenerate={onGenerateInlineInsight} onRefresh={onRefreshInlineInsight} /> : null}
            >
              {!singleSection && sections.length > 0 ? sections.map(section => {
                const sectionTitle = labelOrId(section.title, section.section_id);
                const sectionInsight = getSectionInlineInsight?.(chapter, section);
                return <NavigationTreeItem
                  key={section.section_id}
                  itemId={`section:${section.section_id}`}
                  className="section-item hierarchy-section-item"
                  contentClassName="hierarchy-section-row hierarchy-section-button"
                  label={sectionTitle}
                  disabled={!section.task_units?.length}
                  copy={<SectionLabel section={section} />}
                  action={<SectionInteractionButton chapter={chapter} section={section}
                    sectionTitle={sectionTitle} onOpen={openSectionInteractionMenu} />}
                  insight={sectionInsight ? <InlineInsightRegion className="hierarchy-section-inline-insight"
                    insight={sectionInsight} onDismiss={onDismissInlineInsight}
                    onGenerate={onGenerateInlineInsight} onRefresh={onRefreshInlineInsight} /> : null}
                />;
              }) : null}
            </NavigationTreeItem>
          );
        })}
      </SimpleTreeView>
      <Menu
        className="hierarchy-chapter-interaction-menu"
        id="chapter-interaction-menu"
        anchorEl={chapterInteractionMenu?.anchorEl || null}
        open={chapterInteractionMenuOpen}
        onClose={closeChapterInteractionMenu}
      >
        {summaryNoteMenuItems(selectChapterInteraction)}
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
        {summaryNoteMenuItems(selectSectionInteraction)}
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
