import { Box, Button, Chip, Stack, Typography } from "@mui/material";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ContentBlock, RequestStatus, SectionSelection } from "../../types/api";
import { StateView } from "../../shared/components/StateView";
import {
  headingFromSelection,
  paginateMeasuredGroups,
  type ReaderContentGroup,
  type ReaderContentPage,
} from "./model";

interface ReaderContentViewProps {
  layoutStatus: RequestStatus;
  contentStatus: RequestStatus;
  selectedSection: SectionSelection | null;
  contentBlocks: ContentBlock[];
  contentGroups: ReaderContentGroup[];
  error: string;
}

export function ReaderContentView({
  layoutStatus,
  contentStatus,
  selectedSection,
  contentBlocks,
  contentGroups,
  error,
}: ReaderContentViewProps) {
  const pageListRef = useRef<HTMLDivElement | null>(null);
  const measurementRef = useRef<HTMLDivElement | null>(null);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [pages, setPages] = useState<ReaderContentPage[]>([]);

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

  const visibleBlocks = activeGroups.length > 0
    ? activeGroups.flatMap((group) => group.blocks)
    : contentBlocks;
  const pageCount = pages.length || (contentStatus === "success" ? 1 : 0);
  const displayPageIndex = Math.min(currentPageIndex, Math.max(0, pageCount - 1));
  const canGoPrevious = displayPageIndex > 0;
  const canGoNext = displayPageIndex + 1 < pageCount;

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
        <StateView
          className="reader-content-blocks-error-state"
          title="Section content request failed"
          detail={error}
          severity="error"
        />
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
            {visibleBlocks.map((block, index) => (
              <ContentBlockView block={block} index={index} key={`${block.block_id}:${index}`} />
            ))}
          </Stack>
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
                {group.blocks.map((block, index) => (
                  <ContentBlockView
                    block={block}
                    index={index}
                    key={`${group.taskUnitId}:${block.block_id}:${index}`}
                  />
                ))}
              </Stack>
            ))}
          </Box>
        </>
      ) : null}
    </Box>
  );
}

function ContentBlockView({ block, index }: { block: ContentBlock; index: number }) {
  return (
    <Box
      component="section"
      className="content-block reader-content-block"
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
      </Stack>
      <Typography className="reader-content-block-body" variant="body1">
        {block.content}
      </Typography>
    </Box>
  );
}
