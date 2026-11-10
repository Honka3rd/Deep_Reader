"""Explicit reading overlays. Original hierarchy/source text is never rewritten."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from shared.task_artifacts import DocumentTaskArtifacts

STATE_KEY = "reading_content"
MAX_CHARS = 10000


class ReadingContentConflict(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def source_token_for(document):
    return digest([document.document_id, document.parse_provenance.get("storage_structure_version"),
                   document.parse_provenance.get("interaction_epoch"), document.raw_text,
                   [document.parse_provenance.get(key) for key in (
                       "extraction", "pdf_extraction", "source_review", "source_page_spans")],
                   [(c.chapter_id,c.title,[(s.section_id,s.title,[(u.unit_id,u.title,u.container_title,u.content,
                       [(b.block_id,b.content,b.metadata) for b in u.content_blocks]) for u in s.task_units])
                     for s in c.sections]) for c in document.chapters]])


def state_for(document):
    return document.parse_provenance.get(STATE_KEY,{})


def block_was_manually_edited(state, block_id):
    """Durable server flag; legacy audit fallback is read-only and survives Undo."""
    return bool(state.get("block_edit_flags", {}).get(block_id)) or any(
        event.get("action") == "edit" and event.get("block_id") == block_id
        for event in state.get("audit", [])) or (
            block_id in state.get("blocks", {}) and not state["blocks"][block_id].get("deleted")
            and state["blocks"][block_id].get("origin") != "source_review")


def advance_layout_revision(metadata, expected_revision, revision):
    """Carry only a current revision stamp across explicit content mutations."""
    metadata = dict(metadata)
    layout = metadata.get("task_layout")
    if isinstance(layout, dict) and layout.get("reading_revision") == expected_revision:
        metadata["task_layout"] = {**layout, "reading_revision": revision}
    return metadata


def review_regions(document):
    state = state_for(document)
    decisions = {**state.get("regions", {}), **dict.fromkeys(state.get("permanently_ignored_regions", {}), True)}
    return [{**r,"status":("resolved" if decisions.get(r["id"]) == "cleanup_resolved" else
                           "discarded" if decisions.get(r["id"]) else "unresolved")}
            for r in document.parse_provenance.get("source_review",{}).get("regions",[])]


def needs_review(document):
    from document_structure.source_review_blocks import needs_review_from_provenance, review_provenance
    return needs_review_from_provenance(review_provenance(document))


def pending_review_regions(document):
    """Read-only eligibility shared with Library and interaction generation."""
    from document_structure.source_review_blocks import active_review_regions, review_provenance
    return [{**r, "status": "unresolved"} for r in active_review_regions(review_provenance(document))]


def cleanup_audit_for(document, offset=0):
    from document_structure.cleanup_history import history_for
    return history_for(document, offset)


def original_blocks(document):
    from section_tasks.reading_target_resolver import ReadingTargetResolver
    resolver = ReadingTargetResolver()
    return [(c,s,u,b) for c in document.chapters for s in c.sections for u in s.task_units
            for b in resolver.content_blocks_for_target_validation(u,s,c)]


def projection_source(document):
    state = state_for(document)
    has_spans = bool(state.get("cleanup_spans") or state.get("source_edits"))
    from document_structure.ocr_cleanup_overlay import canonical_text
    text = canonical_text(document) if has_spans else ""
    if has_spans and state.get("cleanup_source_sha256") != hashlib.sha256(text.encode()).hexdigest():
        raise ReadingContentConflict("cleanup_source_changed")
    return state, has_spans, text


def project_task_unit(unit, section, chapter, *, state, text, unit_span=None):
    """Shared body projection for full reads and bounded changed-unit refreshes."""
    from document_structure.ocr_cleanup_overlay import reading_slice, block_range
    from document_structure.text_normalization import normalize_ocr_text
    from section_tasks.reading_target_resolver import ReadingTargetResolver
    blocks = ReadingTargetResolver().content_blocks_for_target_validation(unit, section, chapter)
    updated = []
    for block in blocks:
        edit = state.get("blocks", {}).get(block.block_id)
        if edit and edit.get("deleted"):
            continue
        content = block.content
        if unit_span is not None:
            span = block_range(unit, block, unit_span)
            content = normalize_ocr_text(reading_slice(text, span.start, span.end, state))
        if edit:
            content = edit["content"]
        if content.strip():
            updated.append(replace(block, content=content))
    content = "\n\n".join(b.content for b in updated)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    updated = [replace(b, metadata={**(b.metadata or {}), "reading_overlay": True,
                "source_hash": content_hash, "reading_revision": state.get("revision", 0)}) for b in updated]
    return replace(unit, content=content, content_blocks=updated, task_artifacts=None)


def project_document(document):
    """Pure reading-only projection; never hand this document to a persistence writer."""
    state = state_for(document)
    if (not state.get("blocks") and not state.get("cleanup_spans") and not state.get("source_edits")
            or document.parse_provenance.get("reading_projected")):
        return document
    from document_structure.ocr_cleanup_overlay import reading_slice, unit_ranges
    state, has_spans, text = projection_source(document)
    chapters = []
    for chapter in document.chapters:
        sections = []
        for section in chapter.sections:
            units = []
            ranges = unit_ranges(section) if has_spans else {}
            for unit in section.task_units:
                units.append(project_task_unit(unit, section, chapter, state=state, text=text,
                    unit_span=ranges.get(unit.unit_id)))
            sections.append(replace(section,task_units=units,
                content="\n\n".join(u.content for u in units) if units else (
                    reading_slice(text, section.char_start, section.char_end, state) if has_spans else section.content),
                task_artifacts=None))
        chapters.append(replace(chapter,sections=sections,task_artifacts=None))
    return replace(document,chapters=chapters,parse_provenance={**document.parse_provenance,"reading_projected":True})


class ReadingContentService:
    def __init__(self,repository,loader_factory=None):
        self.repository = repository
        self.loader_factory = loader_factory

    def snapshot(self,document):
        from document_structure.source_review_blocks import region_block_mapping, region_manual_change
        from document_structure.ocr_review_availability import ocr_review_available
        state = state_for(document)
        edits = state.get("blocks",{})
        projected = {b.block_id: b for _,_,_,b in original_blocks(project_document(document))}
        pending = pending_review_regions(document)
        mappings = region_block_mapping(document) if pending else {}
        from document_structure.source_region_repair import region_repair_descriptions
        repairs = region_repair_descriptions(document) if pending else {}
        regions = []
        for region in pending:
            mapping = {**mappings[region["id"]], "block_ids": [key for key in mappings[region["id"]]["block_ids"]
                if key not in state.get("review_kept_blocks", {})]}
            regions.append({**region, **mapping, "repair": repairs.get(region["id"]),
                "related_deleted_count": sum(key not in projected for key in mapping["block_ids"]),
                "manual_change": region_manual_change(state, mapping["block_ids"])})
        result = {"document_id":document.document_id,"revision":state.get("revision",0),
                "structure_version": str(document.parse_provenance["storage_structure_version"])
                    if document.parse_provenance.get("storage_structure_version") is not None else None,
                "source_token":source_token_for(document),"max_chars":MAX_CHARS,
                "disposition":"needs_review" if pending else "completed",
                "regions":regions,"can_undo":bool(state.get("undo")),
                "ocr_review_available": ocr_review_available(document),
                "undo_action":next((e.get("action") for e in reversed(state.get("audit", []))
                                    if e.get("action") not in {"cleanup_edit", "cleanup_delete", "region_edit", "review_block_keep", "discard_all_regions"}), None) if state.get("undo") else None,
                "blocks":[{"block_id":b.block_id,"task_unit_id":u.unit_id,"section_id":s.section_id,
                           "chapter_id":c.chapter_id,"content":edits.get(b.block_id,{}).get("content",
                               projected[b.block_id].content if b.block_id in projected else b.content),
                           "deleted":edits.get(b.block_id,{}).get("deleted",False) or b.block_id not in projected,
                           "modified":b.block_id in edits,
                           "manually_edited":block_was_manually_edited(state, b.block_id)} for c,s,u,b in original_blocks(document)]}
        from document_structure.source_review_block_repair import block_repair_descriptions
        capabilities = block_repair_descriptions(document, result["blocks"], mappings=mappings)
        for block in result["blocks"]:
            block["review_repair"] = capabilities[block["block_id"]]
        return result

    def read(self,doc_name,*,include_cleanup_audit=False,cleanup_offset=0,region_id=None,block_offset=0):
        self.validate_name(doc_name)
        if type(cleanup_offset) is not int or not 0 <= cleanup_offset <= 1000:
            raise ValueError("invalid_cleanup_offset")
        document = self.repository.load_document(doc_name)
        result = self.snapshot(document)
        if region_id is not None:
            if type(block_offset) is not int or block_offset < 0:
                raise ValueError("invalid_block_offset")
            region = next((r for r in result["regions"] if r["id"] == region_id), None)
            if region is None:
                raise FileNotFoundError("review_region_missing")
            ids = set(region["block_ids"])
            blocks = [b for b in result["blocks"] if b["block_id"] in ids and not b["deleted"]]
            result["block_page"] = {"total": len(blocks), "offset": block_offset, "limit": 20}
            result["blocks"] = blocks[block_offset:block_offset + 20]
        if include_cleanup_audit:
            result["cleanup_audit"] = cleanup_audit_for(document, cleanup_offset)
        return result

    def rebuild_review_mapping(self, doc_name, *, expected_revision, source_token):
        """Explicit internal legacy backfill; never called from a read or projection."""
        from document_structure.source_review_blocks import persist_region_mapping
        self.validate_name(doc_name)

        def mutation(document):
            if (source_token != source_token_for(document)
                    or expected_revision != state_for(document).get("revision", 0)):
                raise ReadingContentConflict("review_mapping_changed_reload_required")
            updated = persist_region_mapping(document)
            return updated, self.snapshot(updated)

        return self.repository.mutate_reading_content(doc_name, mutation)

    def read_refresh(self, doc_name, *, mutation_id, expected_revision, source_token, block_offset=0):
        from document_structure.reading_refresh import mutation_refresh
        self.validate_name(doc_name)
        if (type(expected_revision) is not int or expected_revision < 0
                or not isinstance(source_token, str) or len(source_token) != 64
                or not isinstance(mutation_id, str) or not 1 <= len(mutation_id) <= 100
                or type(block_offset) is not int or block_offset < 0):
            raise ValueError("invalid_reading_refresh_query")
        document = self.repository.load_document(doc_name)
        if (state_for(document).get("revision", 0) != expected_revision
                or source_token_for(document) != source_token):
            raise ReadingContentConflict("reading_content_changed_reload_required")
        result = mutation_refresh(document, mutation_id, block_offset=block_offset)
        if result is None:
            raise ReadingContentConflict("reading_refresh_unavailable")
        if not result["scope_current"]:
            raise ReadingContentConflict("reading_content_changed_reload_required")
        return result

    @staticmethod
    def validate_name(doc_name):
        if not doc_name.strip() or doc_name in {".",".."} or any(c in doc_name for c in ("/","\\","\x00")):
            raise ValueError("invalid_document_name")

    def mutate(self,doc_name,*,action,expected_revision,source_token: str,mutation_id,
               block_id=None,text=None,region_id=None,discard_interactions=None,cleanup_record_id=None):
        self.validate_name(doc_name)
        if action not in {"edit","delete","undo","discard_region","discard_all_regions","cleanup_edit","cleanup_restore","cleanup_delete",
                          "region_edit", "region_apply", "region_delete",
                          "review_block_edit", "review_block_delete", "review_block_keep"}:
            raise ValueError("unsupported_reading_action")
        if not mutation_id or len(mutation_id)>100 or type(expected_revision) is not int or expected_revision<0:
            raise ValueError("invalid_reading_revision_or_mutation")
        if action in {"edit", "review_block_edit"} and (not isinstance(text,str) or not text.strip() or len(text)>MAX_CHARS):
            raise ValueError("reading_text_requires_1_to_10000_characters")
        if discard_interactions is not None and type(discard_interactions) is not bool:
            raise ValueError("invalid_discard_interactions")
        cleanup_action = action in {"cleanup_edit", "cleanup_restore", "cleanup_delete"}
        region_action = action in {"region_edit", "region_apply", "region_delete"}
        review_block_action = action in {"review_block_edit", "review_block_delete", "review_block_keep"}
        history_only = action in {"cleanup_edit", "cleanup_delete", "region_edit", "review_block_keep", "discard_all_regions"}
        if action == "discard_all_regions" and any(value is not None for value in (
                block_id, region_id, text, discard_interactions, cleanup_record_id)):
            raise ValueError("invalid_discard_all_regions_command")
        if review_block_action:
            if (not isinstance(block_id, str) or not 1 <= len(block_id) <= 200
                    or not isinstance(region_id, str) or not 1 <= len(region_id) <= 100
                    or cleanup_record_id is not None
                    or action == "review_block_edit" and type(discard_interactions) is not bool
                    or action != "review_block_edit" and (text is not None or discard_interactions is not None)):
                raise ValueError("invalid_review_block_command")
        if region_action:
            if (not isinstance(region_id, str) or not 1 <= len(region_id) <= 100
                    or block_id is not None or cleanup_record_id is not None):
                raise ValueError("invalid_region_command")
            if action == "region_edit" or text is not None:
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
                    raise ValueError("reading_text_requires_1_to_10000_characters")
            if (action == "region_apply" and type(discard_interactions) is not bool
                    or action != "region_apply" and discard_interactions is not None
                    or action == "region_delete" and text is not None):
                raise ValueError("invalid_region_command")
        if cleanup_action:
            if (not isinstance(cleanup_record_id, str) or len(cleanup_record_id) != 64
                    or any(value is not None for value in (block_id, region_id, discard_interactions))):
                raise ValueError("invalid_cleanup_record_command")
            if action == "cleanup_delete" and text is not None:
                raise ValueError("invalid_cleanup_record_command")
            if action == "cleanup_edit" or text is not None:
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
                    raise ValueError("reading_text_requires_1_to_10000_characters")
        elif cleanup_record_id is not None:
            raise ValueError("invalid_cleanup_record_command")
        command_parts = [action,expected_revision,source_token,block_id,text,region_id]
        if cleanup_action:
            command_parts.append(cleanup_record_id)
        if discard_interactions is not None:
            command_parts.append(discard_interactions)
        command = digest(command_parts)

        def mutation(document):
            state = deepcopy(state_for(document))
            receipts = state.setdefault("receipts",{})
            if mutation_id in receipts:
                if receipts[mutation_id] != command:
                    raise ReadingContentConflict("reading_mutation_conflict")
                from document_structure.reading_refresh import mutation_refresh
                return document, {**self.snapshot(document), "refresh": mutation_refresh(document, mutation_id)}
            if source_token != source_token_for(document) or expected_revision != state.get("revision",0):
                raise ReadingContentConflict("reading_content_changed_reload_required")
            previous = {key: deepcopy(state.get(key, default)) for key, default in (
                ("blocks", {}), ("regions", {}), ("source_edits", []), ("cleanup_spans", []),
                ("cleanup_record_applications", {}), ("region_applications", {}))}
            cleanup_affected = []
            region_cleared = []
            ignored_regions = []
            if review_block_action:
                from document_structure.source_review_block_repair import validate_review_block
                validate_review_block(document, block_id, region_id, action)
            if region_action:
                from document_structure.source_region_repair import change_region
                cleanup_affected, region_cleared = change_region(document, state, action=action,
                    region_id=region_id, text=text, discard_interactions=discard_interactions)
            elif cleanup_action:
                from document_structure.cleanup_history import change_record
                cleanup_affected, cleanup_batch_revision = change_record(document, state, action=action, identity=cleanup_record_id,
                                                text=text, previous=previous)
            elif action == "review_block_keep":
                from document_structure.source_review_block_repair import keep_review_block
                keep_review_block(state, block_id)
            elif action in {"edit", "delete", "review_block_edit", "review_block_delete"}:
                matches = [b for _,_,_,b in original_blocks(document) if b.block_id==block_id]
                if len(matches)!=1:
                    raise ReadingContentConflict("reading_block_missing")
                prior = state.setdefault("blocks",{}).get(block_id,{})
                if prior.get("deleted") or not any(b.block_id == block_id for _,_,_,b in original_blocks(project_document(document))):
                    raise ReadingContentConflict("reading_block_deleted_use_undo")
                if region_id is not None:
                    from document_structure.source_review_blocks import region_block_mapping
                    mapping = region_block_mapping(document).get(region_id, {})
                    if (block_id not in mapping.get("block_ids", [])
                            or not any(r["id"] == region_id for r in pending_review_regions(document))):
                        raise ReadingContentConflict("review_block_mapping_changed")
                if action in {"edit", "review_block_edit"}:
                    effective = next(b.content for _, _, _, b in original_blocks(project_document(document))
                                     if b.block_id == block_id)
                    if text == effective:
                        receipts[mutation_id] = command
                        state.setdefault("no_op_mutations", {})[mutation_id] = {
                            "revision": expected_revision, "source_token": source_token}
                        updated = replace(document, parse_provenance={**document.parse_provenance, STATE_KEY: state})
                        from document_structure.reading_refresh import mutation_refresh
                        return updated, {**self.snapshot(updated), "refresh": mutation_refresh(updated, mutation_id)}
                state["blocks"][block_id] = {"content":text if action in {"edit", "review_block_edit"} else prior.get("content",matches[0].content),
                                             "deleted":action in {"delete", "review_block_delete"}}
                if review_block_action:
                    state["blocks"][block_id]["origin"] = "source_review"
                if action == "edit":
                    state.setdefault("block_edit_flags", {})[block_id] = True
            elif action=="discard_region":
                if not any(r["id"]==region_id for r in pending_review_regions(document)):
                    raise ReadingContentConflict("review_region_missing")
                state.setdefault("regions",{})[region_id] = True
            elif action == "discard_all_regions":
                from document_structure.ocr_review_availability import ocr_review_available
                if not ocr_review_available(document):
                    raise ReadingContentConflict("ocr_review_unavailable")
                ignored_regions = [r["id"] for r in pending_review_regions(document)]
                if not ignored_regions:
                    raise ReadingContentConflict("review_regions_missing")
                state.setdefault("permanently_ignored_regions", {}).update(dict.fromkeys(ignored_regions, mutation_id))
                state.setdefault("regions", {}).update(dict.fromkeys(ignored_regions, True))
            else:
                undo = state.get("undo")
                if not undo:
                    raise ReadingContentConflict("nothing_to_undo")
                state.update(undo)
                state.setdefault("regions", {}).update(dict.fromkeys(state.get("permanently_ignored_regions", {}), True))
            from document_structure.reading_refresh import changed_blocks
            affected_body = changed_blocks(document, replace(document, parse_provenance={
                **document.parse_provenance, STATE_KEY: state})) if not history_only else []
            body_changed = bool(affected_body)
            if not history_only and not body_changed and not original_blocks(document):
                before_projection = project_document(document)
                after_projection = project_document(replace(document, parse_provenance={
                    **document.parse_provenance, STATE_KEY: state}))
                body_changed = ([s.content for c in before_projection.chapters for s in c.sections]
                                != [s.content for c in after_projection.chapters for s in c.sections])
            if not history_only and (body_changed or action in {"undo", "discard_region"}):
                state["undo"] = None if action=="undo" else previous
            state["revision"] = expected_revision + (0 if not body_changed and action in {
                "cleanup_restore", "region_apply"} else 1)
            receipts[mutation_id] = command
            affected = [block_id] if action in {"edit", "delete", "review_block_edit", "review_block_delete"} else cleanup_affected
            if action == "undo":
                from document_structure.reading_refresh import changed_blocks
                affected = changed_blocks(document, replace(document, parse_provenance={
                    **document.parse_provenance, STATE_KEY: state}))
            clear_block = block_id if action in {"delete", "review_block_delete"} or (action in {"edit", "review_block_edit"} and discard_interactions is True) else None
            clear_blocks = list(dict.fromkeys(([clear_block] if clear_block is not None else []) + region_cleared))
            for cleared in clear_blocks:
                # Not part of Undo: old note editors must never resurrect purged data.
                state.setdefault("block_interaction_epochs", {})[cleared] = mutation_id
            for changed in affected_body:
                state.setdefault("block_content_epochs", {})[changed] = mutation_id
            invalidate = body_changed and not (action in {"edit", "review_block_edit", "region_apply"}
                                                     and discard_interactions is False)
            state.setdefault("audit",[]).append({"action":action,"block_id":block_id,"region_id":region_id,
                "affected_block_ids":affected,"mutation_id":mutation_id,
                "ignored_regions":ignored_regions,"invalidate_generated":invalidate,"cleared_block_id":clear_block,
                "body_changed": body_changed, "summaries_invalidated": body_changed,
                "translation_block_ids": affected_body,
                "refresh_source_token": source_token,
                "revision":state["revision"],"at":datetime.now(timezone.utc).isoformat()})
            if region_action:
                state["audit"][-1]["cleared_block_ids"] = clear_blocks
            if cleanup_action:
                state["audit"][-1]["cleanup_record_id"] = cleanup_record_id
                state["audit"][-1]["cleanup_batch_revision"] = cleanup_batch_revision
            from document_structure.reading_content_artifacts import apply_artifact_policy
            updated = apply_artifact_policy(document, invalidate_generated=invalidate, cleared_block_id=clear_block,
                                            changed_block_ids=affected_body)
            for cleared in region_cleared:
                updated = apply_artifact_policy(updated, invalidate_generated=False, cleared_block_id=cleared)
            artifacts = updated.document_task_artifacts or DocumentTaskArtifacts()
            # Explicit overlays keep canonical units/IDs. Advance only an already
            # current layout stamp; never bless a missing/stale layout on a read.
            metadata = advance_layout_revision(artifacts.metadata, expected_revision, state["revision"])
            updated = replace(updated,parse_provenance={**document.parse_provenance,STATE_KEY:state},
                              document_task_artifacts=replace(artifacts,metadata=metadata))
            from document_structure.source_review_blocks import persist_region_mapping
            updated = persist_region_mapping(updated)
            if action == "cleanup_restore" or region_action:
                project_document(updated)
            from document_structure.reading_refresh import mutation_refresh
            result = self.snapshot(updated)
            return updated, {**result, "refresh": mutation_refresh(updated, mutation_id)}

        return self.repository.mutate_reading_content(doc_name,mutation)

    def preview(self,doc_name,region_id):
        self.validate_name(doc_name)
        document = self.repository.load_document(doc_name)
        review = document.parse_provenance.get("source_review",{})
        region = next((r for r in pending_review_regions(document) if r["id"]==region_id),None)
        if region is None:
            raise FileNotFoundError("review_region_missing")
        loader = self.loader_factory.get(doc_name)
        preview = getattr(loader,"render_source_region",None)
        if not callable(preview):
            raise ReadingContentConflict("source_preview_unavailable")
        return preview(doc_name,review["source_sha256"],region)
