import { Box, Chip, Stack, Typography } from "@mui/material";
import type { ContentBlock, RequestStatus, SectionSelection } from "../types/api";
import { StateView } from "./StateView";

interface ReaderContentProps {
  layoutStatus: RequestStatus;
  contentStatus: RequestStatus;
  selectedSection: SectionSelection | null;
  contentBlocks: ContentBlock[];
  error: string;
}

function headingFromSelection(selection: SectionSelection) {
  return selection.section.title?.trim() || selection.section.section_id;
}

export function ReaderContent({
  layoutStatus,
  contentStatus,
  selectedSection,
  contentBlocks,
  error,
}: ReaderContentProps) {
  if (layoutStatus === "initial") {
    return <StateView title="Ready" detail="Enter an existing document name to load reading units." />;
  }

  if (layoutStatus === "loading") {
    return <StateView title="Loading document" loading />;
  }

  if (layoutStatus === "error") {
    return <StateView title="No content loaded" />;
  }

  if (!selectedSection) {
    return <StateView title="No section selected" detail="Choose a section from the hierarchy." />;
  }

  return (
    <Box>
      <Box component="header" className="content-header">
        <Typography className="eyebrow" color="text.secondary">
          {[selectedSection.section.container_title, selectedSection.chapter.title]
            .filter(Boolean)
            .join(" / ")}
        </Typography>
        <Typography component="h1" variant="h1">
          {headingFromSelection(selectedSection)}
        </Typography>
      </Box>

      {contentStatus === "loading" ? <StateView title="Loading content" loading /> : null}
      {contentStatus === "error" ? (
        <StateView title="Section content request failed" detail={error} severity="error" />
      ) : null}
      {contentStatus === "empty" ? <StateView title="No content blocks" /> : null}

      {contentStatus === "success" ? (
        <Stack component="article" spacing={2.25} className="content-blocks">
          {contentBlocks.map((block, index) => (
            <Box component="section" className="content-block" key={`${block.block_id}:${index}`}>
              <Stack direction="row" spacing={1} alignItems="center" className="block-meta">
                <Chip size="small" label={block.block_type || "content"} />
                <Typography variant="caption" color="text.secondary">
                  {block.block_id}
                </Typography>
              </Stack>
              <Typography variant="body1">{block.content}</Typography>
            </Box>
          ))}
        </Stack>
      ) : null}
    </Box>
  );
}
