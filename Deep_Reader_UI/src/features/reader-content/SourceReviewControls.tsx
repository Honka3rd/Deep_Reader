import { useEffect, useState } from "react";
import { IconButton, ListItemIcon, Menu, MenuItem, Tooltip } from "@mui/material";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import CheckIcon from "@mui/icons-material/Check";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";

export type SourceReviewAction = "edit" | "delete" | "keep";

export function SourceReviewItemMenu({ identity, label, classPrefix, disabled, capabilities, hints, onSelect }: {
  identity: string; label: string; classPrefix: string; disabled: boolean;
  capabilities: Record<SourceReviewAction, boolean>;
  hints?: Partial<Record<SourceReviewAction, string>>;
  onSelect: (action: SourceReviewAction, trigger: HTMLElement) => void;
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  useEffect(() => { setAnchor(null); }, [identity, disabled]);
  return <>
    <Tooltip title="Text actions"><span>
      <IconButton className={`source-review-item-menu-trigger ${classPrefix}-menu-trigger`} aria-label={label}
        aria-haspopup="menu" aria-expanded={Boolean(anchor)} disabled={disabled} onClick={event => setAnchor(event.currentTarget)}>
        <MoreVertIcon />
      </IconButton>
    </span></Tooltip>
    <Menu className={`source-review-item-menu ${classPrefix}-menu`} anchorEl={anchor} open={Boolean(anchor)}
      onClose={() => setAnchor(null)}>
      {(["edit", "delete", "keep"] as const).map(action => <MenuItem key={action}
        className={`${classPrefix}-${action}-action`} title={hints?.[action]}
        disabled={disabled || !capabilities[action]}
        onClick={() => { if (anchor) onSelect(action, anchor); setAnchor(null); }}>
        <ListItemIcon>{action === "edit" ? <EditOutlinedIcon fontSize="small" />
          : action === "delete" ? <DeleteOutlineIcon fontSize="small" /> : <CheckIcon fontSize="small" />}</ListItemIcon>
        {action === "edit" ? "Edit" : action === "delete" ? "Delete" : "Keep it"}
      </MenuItem>)}
    </Menu>
  </>;
}

export function SourceReviewNavigateButton({ blockId, disabled, onClick }: {
  blockId: string; disabled: boolean; onClick: () => void;
}) {
  return <Tooltip title="Go to content"><span>
    <IconButton className="source-review-navigate-action" aria-label={`Go to content for block ${blockId}`}
      disabled={disabled} onClick={onClick}><ArrowForwardIcon /></IconButton>
  </span></Tooltip>;
}
