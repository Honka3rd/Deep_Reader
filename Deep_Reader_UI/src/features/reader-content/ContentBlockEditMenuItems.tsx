import { ListItemIcon, ListItemText, MenuItem } from "@mui/material";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";

export function contentBlockEditMenuItems(onSelect: (action: "edit" | "delete") => void) {
  return (["edit", "delete"] as const).map(action => <MenuItem key={action}
    className={`reader-content-block-${action}-action`} onClick={() => onSelect(action)}>
    <ListItemIcon>{action === "edit" ? <EditOutlinedIcon fontSize="small" /> : <DeleteOutlineIcon fontSize="small" />}</ListItemIcon>
    <ListItemText primary={action === "edit" ? "Edit" : "Delete"} />
  </MenuItem>);
}
