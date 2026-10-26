import { Box, Button, Chip, CircularProgress, IconButton, Stack, Tooltip, Typography } from "@mui/material";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import RefreshIcon from "@mui/icons-material/Refresh";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { useSourceReviewBlocks, type ReviewNavigate } from "./useSourceReviewBlocks";
import { canNavigateReviewBlock, manualChangeLabel, reviewMappingMessage } from "./sourceReviewModel";
import type { ReadingBlock, ReadingState, ReviewRegion } from "../../services/ReadingContentService";
import { SourceReviewBlockMenu, reviewBlockReason, type ReviewBlockMode } from "./SourceReviewBlockRepair";
import { SourceReviewNavigateButton } from "./SourceReviewControls";

export function SourceReviewManualChange({ region }: { region: ReviewRegion }) {
  const change = manualChangeLabel(region);
  return change && <Stack className="source-review-manual-change" direction="row" spacing={1} alignItems="center" flexWrap="wrap">
    <Chip size="small" label={change.label} />
    {change.at && <Typography component="time" dateTime={region.manual_change!.at} variant="caption">{change.at}</Typography>}
  </Stack>;
}

export function SourceReviewBlocks({ docName, region, snapshot, activation, disabled, onSnapshot, onNavigate, onRepair }: {
  docName: string; region: ReviewRegion; snapshot: ReadingState; activation: number; disabled: boolean;
  onSnapshot: (next: ReadingState) => void; onNavigate: ReviewNavigate;
  onRepair?: (block: ReadingBlock, mode: ReviewBlockMode, trigger: HTMLElement) => void;
}) {
  const { state, error, busy, offset, setOffset, reload, navigate } = useSourceReviewBlocks(
    docName, region.id, activation, snapshot, onSnapshot, onNavigate);
  const currentRegion = state?.regions.find(item => item.id === region.id);
  const total = state?.block_page?.total ?? state?.blocks.length ?? 0;
  const limit = state?.block_page?.limit || 20;
  const message = currentRegion ? reviewMappingMessage(currentRegion) : "";
  const first = currentRegion && state?.blocks.find(block => canNavigateReviewBlock(currentRegion, block));
  return <Box className="source-review-blocks" data-region-id={region.id} aria-busy={busy}>
    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
      <Button className="source-review-go-to-content" startIcon={<ArrowForwardIcon />}
        disabled={disabled || busy || !first} onClick={() => first && navigate(first.block_id)}>Go to content</Button>
      <Tooltip title="Refresh related blocks"><span><IconButton aria-label="Refresh related blocks" disabled={disabled || busy} onClick={reload}><RefreshIcon /></IconButton></span></Tooltip>
    </Stack>
    {error && <Typography role="status">{error}</Typography>}
    {busy && <CircularProgress size={20} aria-label="Loading region blocks" />}
    {message && <Typography className="source-review-blocks-unavailable" variant="body2" color="text.secondary">{message}</Typography>}
    {currentRegion && (currentRegion.related_deleted_count ?? 0) > 0 && <Typography variant="caption">
      Deleted related blocks: {currentRegion.related_deleted_count}
    </Typography>}
    {state && <Box className="source-review-block-list" sx={{ maxHeight: 480, overflow: "auto" }}>
      {state.blocks.map(block => <Box key={block.block_id} className="source-review-block" data-block-id={block.block_id}
        sx={{ borderTop: "1px solid", borderColor: "divider", py: 1 }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
          <Typography className="source-review-block-id" variant="caption" sx={{ overflowWrap: "anywhere", minWidth: 0, flex: 1 }}>{block.block_id}</Typography>
          {onRepair && <SourceReviewBlockMenu block={block} disabled={disabled || busy}
            onSelect={(mode, trigger) => onRepair(block, mode, trigger)} />}
          <SourceReviewNavigateButton blockId={block.block_id}
            disabled={disabled || busy || !currentRegion || !canNavigateReviewBlock(currentRegion, block)}
            onClick={() => navigate(block.block_id)} />
        </Stack>
        {block.deleted && <Typography variant="caption">Deleted</Typography>}
        {block.review_repair?.blocked_reason && <Typography className="source-review-block-protection" variant="caption">{reviewBlockReason(block)}</Typography>}
        <Typography className="source-review-block-content" sx={{ whiteSpace: "pre-wrap", maxHeight: 220, overflow: "auto", overflowWrap: "anywhere" }}>{block.content}</Typography>
      </Box>)}
      {total === 0 && <Typography>No current related blocks.</Typography>}
    </Box>}
    {state && total > limit && <Stack className="source-review-block-pagination" direction="row" alignItems="center">
      <IconButton aria-label="Previous region blocks" disabled={busy || offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}><ChevronLeftIcon /></IconButton>
      <Typography>{offset + 1}-{Math.min(offset + limit, total)} / {total}</Typography>
      <IconButton aria-label="Next region blocks" disabled={busy || offset + limit >= total} onClick={() => setOffset(offset + limit)}><ChevronRightIcon /></IconButton>
    </Stack>}
  </Box>;
}
