import type { DocumentTaskLayout } from "../../types/api";
import type { ReadingRefresh, ReadingRefreshBlock } from "../../services/ReadingContentService";
import type { ReaderContentBlock, ReaderContentGroup } from "./model";

const changed = () => new Error("Content changed or refresh is unavailable. Reload before continuing.");

export function validateRefreshPage(page: ReadingRefresh, binding: ReadingRefresh, offset: number) {
  if (!page || !binding || !page.scope_current || !binding.scope_current
    || page.revision !== page.mutation_revision || binding.revision !== binding.mutation_revision
    || ["document_id", "revision", "source_token", "mutation_id", "mutation_revision", "generated_invalidated"]
      .some(key => page[key as keyof ReadingRefresh] !== binding[key as keyof ReadingRefresh])
    || ["structure_version", "cleanup_record_id", "cleared_block_id"].some(key =>
      (page[key as keyof ReadingRefresh] ?? null) !== (binding[key as keyof ReadingRefresh] ?? null))
    || JSON.stringify(page.cleared_block_ids ?? []) !== JSON.stringify(binding.cleared_block_ids ?? [])
    || !Number.isInteger(page.revision) || page.revision < 0 || !page.source_token || !page.mutation_id
    || !page.block_page || !Number.isInteger(page.block_page.total) || page.block_page.total < 0
    || page.block_page.total !== binding.block_page.total || page.block_page.offset !== offset
    || page.block_page.limit !== 20 || !Array.isArray(page.blocks)
    || page.blocks.length !== Math.min(20, Math.max(0, page.block_page.total - offset))) throw changed();
}

export function mergeReadingRefresh(groups: ReaderContentGroup[], layout: DocumentTaskLayout,
  refresh: ReadingRefresh, changes: ReadingRefreshBlock[]): ReaderContentGroup[] {
  if (layout.document_id !== refresh.document_id || changes.length !== refresh.block_page.total
    || new Set(changes.map(row => row.block_id)).size !== changes.length) throw changed();
  const byUnit = new Map<string, ReadingRefreshBlock[]>();
  for (const row of changes) {
    const section = layout.chapters?.find(chapter => chapter.chapter_id === row.chapter_id)?.sections
      ?.find(item => item.section_id === row.section_id);
    if (!section?.task_units?.some(unit => unit.unit_id === row.task_unit_id) || !row.block_id
      || !Number.isInteger(row.block_order) || row.block_order < 0 || !row.unit_source_hash
      || typeof row.manually_edited !== "boolean" || typeof row.deleted !== "boolean"
      || row.deleted !== (row.content_block == null)
      || (row.content_block && (row.content_block.block_id !== row.block_id || typeof row.content_block.content !== "string"))) throw changed();
    const siblings = byUnit.get(row.task_unit_id) ?? [];
    if (siblings.some(other => other.unit_source_hash !== row.unit_source_hash || other.block_order === row.block_order
      || other.chapter_id !== row.chapter_id || other.section_id !== row.section_id)) throw changed();
    siblings.push(row); byUnit.set(row.task_unit_id, siblings);
  }
  const invalidate = (block: ReaderContentBlock) => refresh.generated_invalidated || refresh.cleared_block_id === block.block_id
    || refresh.cleared_block_ids?.includes(block.block_id)
    ? { ...block, interaction_availability: undefined, artifact_target_refs: undefined } : block;
  return groups.map(group => {
    const rows = byUnit.get(group.taskUnitId);
    if (!rows) return refresh.generated_invalidated ? { ...group, blocks: group.blocks.map(invalidate) } : group;
    if (rows[0].chapter_id !== group.chapterId || rows[0].section_id !== group.sectionId) throw changed();
    const affected = new Set(rows.map(row => row.block_id));
    let blocks = group.blocks.filter(block => !affected.has(block.block_id));
    for (const row of [...rows].sort((a, b) => a.block_order - b.block_order)) {
      if (row.deleted) continue;
      const anchor = row.insert_before_block_id;
      const position = anchor == null ? blocks.length : blocks.findIndex(block => block.block_id === anchor);
      if (position < 0 || (anchor != null && affected.has(anchor))) throw changed();
      const previous = group.blocks.find(block => block.block_id === row.block_id);
      const block: ReaderContentBlock = { ...row.content_block!,
        interaction_availability: row.content_block!.interaction_availability ?? previous?.interaction_availability,
        artifact_target_refs: row.content_block!.artifact_target_refs ?? previous?.artifact_target_refs,
        metadata: { ...row.content_block!.metadata, manually_edited: row.manually_edited },
        ownerContext: { documentId: refresh.document_id, documentTitle: previous?.ownerContext.documentTitle ?? layout.title,
          taskUnitId: row.task_unit_id, chapterId: row.chapter_id, sectionId: row.section_id,
          sourceHash: row.unit_source_hash } };
      blocks.splice(position, 0, block);
    }
    blocks = blocks.map(block => invalidate({ ...block,
      metadata: { ...block.metadata, source_hash: rows[0].unit_source_hash, reading_revision: refresh.revision },
      ownerContext: { ...block.ownerContext, sourceHash: rows[0].unit_source_hash } }));
    return { ...group, blocks };
  });
}
