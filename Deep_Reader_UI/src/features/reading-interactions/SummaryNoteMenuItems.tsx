import { ListItemIcon, ListItemText, MenuItem } from "@mui/material";
import SummarizeOutlinedIcon from "@mui/icons-material/SummarizeOutlined";
import NoteAltOutlinedIcon from "@mui/icons-material/NoteAltOutlined";

export function summaryNoteMenuItems(onSelect: (kind: "summary" | "note") => void) {
  return [
    <MenuItem key="summary" onClick={() => onSelect("summary")}><ListItemIcon><SummarizeOutlinedIcon fontSize="small" /></ListItemIcon><ListItemText>Summary</ListItemText></MenuItem>,
    <MenuItem key="note" onClick={() => onSelect("note")}><ListItemIcon><NoteAltOutlinedIcon fontSize="small" /></ListItemIcon><ListItemText>Note</ListItemText></MenuItem>,
  ];
}
