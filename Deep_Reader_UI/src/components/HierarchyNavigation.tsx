import {
  Box,
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
} from "../types/api";
import { StateView } from "./StateView";

interface HierarchyNavigationProps {
  layout: DocumentTaskLayout | null;
  selectedSectionId: string | null;
  status: RequestStatus;
  error: string;
  onSelectSection: (selection: SectionSelection) => void;
}

function countTaskUnits(layout: DocumentTaskLayout | null): number {
  return (
    layout?.chapters?.reduce(
      (chapterTotal, chapter) =>
        chapterTotal +
        (chapter.sections || []).reduce(
          (sectionTotal, section) => sectionTotal + (section.task_units || []).length,
          0,
        ),
      0,
    ) || 0
  );
}

function labelOrId(label: string | null | undefined, fallback: string) {
  return label?.trim() || fallback;
}

function sameDisplayLabel(left: string | null | undefined, right: string | null | undefined) {
  return (
    Boolean(left?.trim()) &&
    left?.trim().toLocaleLowerCase() === right?.trim().toLocaleLowerCase()
  );
}

function isGenericContainerTitle(title: string | null | undefined) {
  const normalizedTitle = title?.trim().toLocaleLowerCase();
  return (
    normalizedTitle === "front matter" ||
    normalizedTitle === "back matter" ||
    normalizedTitle === "table of contents"
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

  return (
    <ButtonBase
      className={selected ? "section-button selected" : "section-button"}
      aria-selected={selected}
      disabled={taskUnits.length === 0}
      onClick={() => onSelectSection({ chapter, section, taskUnits })}
    >
      <Box className="section-button-copy">
        <Typography component="span" variant="body1">
          {resolvedTitle}
        </Typography>
        {displaySubtitle || section.container_title ? (
          <Typography component="span" variant="caption" color="text.secondary">
            {displaySubtitle || section.container_title}
          </Typography>
        ) : null}
      </Box>
      <Chip size="small" label={`${taskUnits.length} units`} />
    </ButtonBase>
  );
}

export function HierarchyNavigation({
  layout,
  selectedSectionId,
  status,
  error,
  onSelectSection,
}: HierarchyNavigationProps) {
  if (status === "initial") {
    return <StateView title="No document loaded" />;
  }

  if (status === "loading") {
    return <StateView title="Loading hierarchy" loading />;
  }

  if (status === "error") {
    return <StateView title="Task layout request failed" detail={error} severity="error" />;
  }

  if (!layout?.chapters?.length || countTaskUnits(layout) === 0) {
    return (
      <StateView
        title="No task units"
        detail="The backend returned no navigable task units for this document."
      />
    );
  }

  return (
    <Box>
      <Stack direction="row" spacing={1} alignItems="center" className="nav-title-row">
        <Typography component="h1" variant="h2">
          {layout.title || "Document"}
        </Typography>
        <Chip size="small" label={`${countTaskUnits(layout)} units`} />
      </Stack>
      <List component="ol" className="chapter-list" disablePadding>
        {layout.chapters.map((chapter) => {
          const sections = chapter.sections || [];
          const singleSection = sections.length === 1 ? sections[0] : null;
          const chapterTitle = labelOrId(chapter.title, chapter.chapter_id);
          const sectionTitle = singleSection
            ? labelOrId(singleSection.title, singleSection.section_id)
            : null;
          const useSectionTitle =
            Boolean(singleSection?.title?.trim()) && isGenericContainerTitle(chapter.title);
          const mergedTitle = useSectionTitle && sectionTitle ? sectionTitle : chapterTitle;
          const mergedSubtitle =
            useSectionTitle && !sameDisplayLabel(chapter.title, singleSection?.title)
              ? chapterTitle
              : singleSection && !sameDisplayLabel(chapter.title, singleSection.title)
              ? sectionTitle
              : singleSection?.container_title || null;

          return (
            <ListItem
              component="li"
              className={singleSection ? "chapter-item merged" : "chapter-item"}
              key={chapter.chapter_id}
            >
              {singleSection ? (
                <SectionButton
                  chapter={chapter}
                  section={singleSection}
                  selected={selectedSectionId === singleSection.section_id}
                  displayTitle={mergedTitle}
                  displaySubtitle={mergedSubtitle}
                  onSelectSection={onSelectSection}
                />
              ) : (
                <>
                  <Typography component="h2" variant="h2">
                    {chapterTitle}
                  </Typography>
                  <List component="ol" className="section-list" disablePadding>
                    {sections.map((section) => (
                      <ListItem component="li" className="section-item" key={section.section_id}>
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
