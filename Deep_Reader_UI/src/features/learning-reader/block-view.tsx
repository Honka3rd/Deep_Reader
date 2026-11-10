import { Fragment, useId, useRef, useState } from "react";
import { Box, Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Stack, Typography } from "@mui/material";
import type { ReaderContentBlock } from "../reader-content/model";
import type { LearningComponent } from "../../services/LearningService";
import { useLearningPresentation } from "./context";
import { HintAssessmentView } from "./HintAssessmentView";
import { componentAction, itemKey, learningHints, learningRows, type LearningItem } from "./model";

const labels = { queued: "Queued", reading: "Reading saved data", pending: "In progress", ready: "Saved", empty: "No learning content",
  partial: "Partially completed", failed: "Failed", unknown: "Outcome unknown", stale: "Source or settings changed", unavailable: "Unavailable", error: "Read failed" };
function ComponentControls({ item }: { item: LearningItem }) {
  const context = useLearningPresentation()!, c = context.controller, key = itemKey(item.blockId, item.component);
  const label = item.component === "translation" ? "translation" : "vocabulary";
  const action = componentAction(item), actionLabel = action === "refresh" ? "Refresh" : action === "retry" || action === "confirm" ? "Retry" : "Check progress for";
  const [confirmation, setConfirmation] = useState<string | null>(null);
  const binding = JSON.stringify([c.identity, key, item.view.operation]);
  const open = confirmation === binding && action === "confirm";
  const cancel = useRef<HTMLButtonElement | null>(null), trigger = useRef<HTMLButtonElement | null>(null), id = useId();
  const invoke = () => {
    if (item.busy || item.phase === "empty") return;
    if (action === "confirm") setConfirmation(binding);
    else if (action === "refresh") c.refresh(key);
    else if (action === "retry") c.retry(key);
    else c.check(key);
  };
  return <Stack className="learning-component-controls" data-component={item.component} gap={.5} sx={{ mb: 1 }}>
    <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
      <Typography className="learning-component-status" variant="caption" role="status">
        {item.component === "translation" ? "Translation" : "Vocabulary"}: {labels[item.phase]}{item.view.saved && item.phase !== "ready" ? " · saved result retained" : ""}
      </Typography>
      <Button ref={trigger} className="learning-component-action" size="small" aria-disabled={!!item.busy || item.phase === "empty"}
        sx={{ opacity: item.busy || item.phase === "empty" ? .5 : 1 }}
        aria-label={`${actionLabel} ${label}`} onClick={invoke}>{actionLabel} {label}</Button>
      {action === "confirm" && <Button className="learning-component-check" size="small" disabled={!!item.busy} onClick={() => c.check(key)}>Check progress</Button>}
    </Stack>
    {item.message && <Typography className="learning-component-message" variant="caption">{item.message}</Typography>}
    <Dialog className="learning-retry-dialog" open={open} onClose={() => setConfirmation(null)} fullWidth maxWidth="xs"
      aria-labelledby={`${id}-title`} aria-describedby={`${id}-description`}
      TransitionProps={{ onEntered: () => cancel.current?.focus(), onExited: () => trigger.current?.focus({ preventScroll: true }) }}>
      <DialogTitle id={`${id}-title`}>Retry {label}?</DialogTitle>
      <DialogContent><DialogContentText id={`${id}-description`}>The previous result is unknown. It may already have incurred a charge.
        Retrying can incur duplicate cost. You can cancel and check progress first.</DialogContentText></DialogContent>
      <DialogActions>
        <Button className="learning-retry-cancel" ref={cancel} autoFocus onClick={() => setConfirmation(null)}>Cancel</Button>
        <Button className="learning-retry-confirm" onClick={() => { if (confirmation !== binding || item.busy) return; setConfirmation(null); c.retry(key, true); }}>Retry with possible duplicate cost</Button>
      </DialogActions>
    </Dialog>
  </Stack>;
}
function useBlock(block: ReaderContentBlock) {
  const context = useLearningPresentation();
  const data = context?.controller.state.blocks[block.block_id];
  // Never apply even a retained projection to another source or render-only block.
  const matches = !!block.block_id?.trim() && !!block.ownerContext.taskUnitId
    && context?.book.doc_name === block.ownerContext.documentId
    && data?.sentences?.map(s => s.text).join("") === block.content;
  return { context, data: matches ? data : undefined };
}
export function LearningBlockBody({ block, highlighted }: { block: ReaderContentBlock; highlighted: boolean }) {
  const { context, data } = useBlock(block);
  const sourceText = (text: string) => highlighted ? <mark className="reader-content-navigation-highlight">{text}</mark> : text;
  if (!context || !data?.sentences) return <>{sourceText(block.content)}</>;
  const item = context.controller.state.items[itemKey(block.block_id, "translation")];
  const projection = learningRows(item), rows = new Map(projection.rows.map(r => [r.sentence_id, r.translated_text]));
  return <Box className="learning-sentence-pairs" sx={{ overflowWrap: "anywhere" }}>
    {data.sentences.map(s => s.sentence_id === null ? <Fragment key={`gap:${s.start}`}><span className="learning-source-gap" style={{ whiteSpace: "pre-wrap" }}>{sourceText(s.text)}</span></Fragment>
      : <Box className="learning-sentence-pair" key={s.sentence_id} data-sentence-id={s.sentence_id} sx={{ mb: 1.5 }}>
        <Box className="learning-source-sentence" lang={context.book.source_language ?? undefined} dir="auto" sx={{ whiteSpace: "pre-wrap", unicodeBidi: "plaintext" }}>{sourceText(s.text)}</Box>
        {rows.has(s.sentence_id) && <Box className="learning-translated-sentence" lang={context.book.workspace.native_language ?? undefined} dir="auto"
          sx={{ mt: .5, pl: 1.5, borderLeft: 2, borderColor: "divider", color: "text.secondary", whiteSpace: "pre-wrap", unicodeBidi: "plaintext" }}>
          {projection.partial && <Typography className="learning-partial-label" component="span" variant="caption">Partial · </Typography>}{rows.get(s.sentence_id)}
        </Box>}
      </Box>)}
  </Box>;
}
export function LearningBlockDetails({ block }: { block: ReaderContentBlock }) {
  const { context, data } = useBlock(block);
  if (!context || !data) return null;
  const state = context.controller.state, translation = state.items[itemKey(block.block_id, "translation")], hints = state.items[itemKey(block.block_id, "hints")];
  const projection = learningHints(hints);
  return <Box className="learning-block-details" sx={{ mt: 1 }}>
    {translation && <ComponentControls item={translation} />}
    <Box className="learning-vocabulary" role="region" aria-label="Vocabulary for this paragraph" sx={{ borderTop: 1, borderColor: "divider", pt: 1 }}>
      {hints && <ComponentControls item={hints} />}
      {projection.partial && <Typography className="learning-partial-label" variant="caption">Partial vocabulary · not yet a saved result</Typography>}
      {projection.assessment && <HintAssessmentView value={projection.assessment} />}
      {projection.saved && !projection.items.length && !projection.assessment?.rejected_count && !projection.assessment?.omitted_sentences
        && <Typography className="learning-vocabulary-empty" variant="body2" color="text.secondary">No vocabulary hints needed for this level.</Typography>}
      <Box className="learning-hint-list" component="dl" sx={{ m: 0, overflowWrap: "anywhere" }}>
        {projection.items.map(h => <Box className="learning-hint" key={JSON.stringify([h.sentence_id, h.span.start, h.span.end, h.kind])} data-sentence-id={h.sentence_id} sx={{ mb: 1 }}>
          <Typography component="dt" variant="body2"><Box component="b" lang={h.language_code} dir="auto">{h.source_form}</Box>
            {h.lemma && h.lemma !== h.source_form && <> · <bdi lang={h.language_code}>{h.lemma}</bdi></>}
            <Typography component="span" variant="caption" color="text.secondary"> · {h.kind}{h.part_of_speech ? ` · ${h.part_of_speech}` : ""}</Typography>
          </Typography>
          <Typography component="dd" variant="body2" lang={context.book.workspace.native_language ?? undefined} dir="auto" sx={{ m: 0, whiteSpace: "pre-wrap", unicodeBidi: "plaintext" }}>{h.meaning}</Typography>
        </Box>)}
      </Box>
    </Box>
  </Box>;
}
