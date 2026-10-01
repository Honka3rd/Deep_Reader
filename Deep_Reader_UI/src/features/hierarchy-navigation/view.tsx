import {
  Box,
  Button,
  ButtonBase,
  Chip,
  Divider,
  List,
  ListItem,
  Stack,
  Typography,
} from "@mui/material";
import type {
  ChapterLayout,
  DocumentTaskLayout,
  RequestStatus,
  SectionLayout,
  SectionSelection,
} from "../../types/api";
import { StateView } from "../../shared/components/StateView";
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

export function HierarchyNavigationView({
  layout,
  selectedSectionId,
  status,
  error,
  canEditToc = false,
  onSelectSection,
  onEditToc,
}: HierarchyNavigationViewProps) {
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
      </Stack>
      <List component="ol" className="chapter-list hierarchy-chapter-list" disablePadding>
        {layout.chapters.map((chapter) => {
          const sections = chapter.sections || [];
          const singleSection = sections.length === 1 ? sections[0] : null;
          const chapterTitle = labelOrId(chapter.title, chapter.chapter_id);
          const mergedSectionDisplay = singleSection
            ? buildSectionDisplay(chapter, singleSection)
            : null;

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
                <SectionButton
                  chapter={chapter}
                  section={singleSection}
                  selected={selectedSectionId === singleSection.section_id}
                  displayTitle={mergedSectionDisplay?.title}
                  displaySubtitle={mergedSectionDisplay?.subtitle}
                  onSelectSection={onSelectSection}
                />
              ) : (
                <>
                  <Typography className="hierarchy-chapter-title" component="h2" variant="h2">
                    {chapterTitle}
                  </Typography>
                  <List
                    component="ol"
                    className="section-list hierarchy-section-list"
                    disablePadding
                  >
                    {sections.map((section) => (
                      <ListItem
                        component="li"
                        className="section-item hierarchy-section-item"
                        key={section.section_id}
                      >
                        <SectionButton
                          chapter={chapter}
                          section={section}
                          selected={selectedSectionId === section.section_id}
                          onSelectSection={onSelectSection}
                        />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
              <Divider />
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
}
