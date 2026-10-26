import type { ReadingBlock, ReadingState, ReviewNavigationBlock, ReviewRegion } from "../../services/ReadingContentService";

export function sameReadingSnapshot(left: ReadingState, right: ReadingState) {
  return left.source_token === right.source_token && left.revision === right.revision
    && (left.structure_version ?? null) === (right.structure_version ?? null);
}

export function matchesReviewDisplayContent(displayed: string, raw: string, normalization: unknown) {
  if (displayed === raw) return true;
  if (normalization !== "ocr_whitespace_v1") return false;
  // Compare only the advertised backend display projection; never change saved text.
  const chars = Array.from(raw.normalize("NFKC"));
  const whitespace = /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]$/u;
  const cjk = (value: string) => /^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]$/u.test(value);
  const punctuation = (value: string) => /^\p{P}$/u.test(value);
  const output: string[] = [];
  for (let index = 0; index < chars.length;) {
    if (!whitespace.test(chars[index])) { output.push(chars[index++]); continue; }
    const previous = chars[index - 1];
    let newlines = 0;
    while (index < chars.length && whitespace.test(chars[index])) {
      if (chars[index] === "\n" || chars[index] === "\r") newlines++;
      index++;
    }
    const following = chars[index];
    if (newlines >= 2) output.push("\n\n");
    else if (!(previous && following && (cjk(previous) || cjk(following)
      || (punctuation(previous) && punctuation(following))))) output.push(" ");
  }
  return displayed === output.join("").replace(/^[ \n]+|[ \n]+$/g, "");
}

export function isReviewNavigationBlock(block: ReadingBlock): block is ReviewNavigationBlock {
  return block.deleted === false && [block.block_id, block.chapter_id, block.section_id, block.task_unit_id]
    .every(id => typeof id === "string" && id.trim().length > 0);
}

export function canNavigateReviewBlock(region: ReviewRegion, block: ReadingBlock) {
  return region.body_status === "linked"
    && (region.mapping_status === "complete" || region.mapping_status === "partial")
    && region.block_ids?.includes(block.block_id) === true && isReviewNavigationBlock(block);
}

export function reviewMappingMessage(region: ReviewRegion) {
  if ((region.related_deleted_count ?? 0) > 0 && !region.block_ids?.length) return "Related content was deleted.";
  if (region.mapping_reason === "blocks_not_materialized" || region.body_status === "pending") return "Related blocks are not available yet.";
  if (region.mapping_reason === "source_mapping_unavailable") return "Source location is unavailable.";
  if (region.mapping_status === "partial") return "Some source text has no verified content link.";
  if (region.body_status !== "linked" || region.mapping_status !== "complete") return "Content mapping unavailable.";
  return "";
}

export function manualChangeLabel(region: ReviewRegion) {
  const change = region.manual_change;
  if (!change || !["modified", "deleted", "restored"].includes(change.status)) return null;
  const date = new Date(change.at);
  return { label: `Manually ${change.status}`, at: Number.isNaN(date.getTime()) ? null : date.toLocaleString() };
}
