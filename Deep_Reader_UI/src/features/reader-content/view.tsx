import { Box, Chip, Stack, Typography } from "@mui/material";
import type { ContentBlock, RequestStatus, SectionSelection } from "../../types/api";
import { StateView } from "../../shared/components/StateView";
import { headingFromSelection } from "./model";

interface ReaderContentViewProps {
  layoutStatus: RequestStatus;
  contentStatus: RequestStatus;
  selectedSection: SectionSelection | null;
  contentBlocks: ContentBlock[];
  error: string;
}

export function ReaderContentView({
  layoutStatus,
  contentStatus,
  selectedSection,
  contentBlocks,
  error,
}: ReaderContentViewProps) {
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
        <Stack
          component="article"
          spacing={2.25}
          className="content-blocks reader-content-block-list"
        >
          {contentBlocks.map((block, index) => (
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
          ))}
        </Stack>
      ) : null}
    </Box>
  );
}
