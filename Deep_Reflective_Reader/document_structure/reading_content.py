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
                   [(c.chapter_id,[(s.section_id,[(u.unit_id,u.content) for u in s.task_units])
                     for s in c.sections]) for c in document.chapters]])


def state_for(document):
    return document.parse_provenance.get(STATE_KEY,{})


def review_regions(document):
    decisions = state_for(document).get("regions",{})
    return [{**r,"status":"discarded" if decisions.get(r["id"]) else "unresolved"}
            for r in document.parse_provenance.get("source_review",{}).get("regions",[])]


def needs_review(document):
    return any(r["status"] == "unresolved" for r in review_regions(document))


def original_blocks(document):
    from section_tasks.reading_target_resolver import ReadingTargetResolver
    resolver = ReadingTargetResolver()
    return [(c,s,u,b) for c in document.chapters for s in c.sections for u in s.task_units
            for b in resolver.content_blocks_for_target_validation(u,s,c)]


def project_document(document):
    """Pure reading-only projection; never hand this document to a persistence writer."""
    state = state_for(document)
    if not state.get("blocks") or document.parse_provenance.get("reading_projected"):
        return document
    from section_tasks.reading_target_resolver import ReadingTargetResolver
    resolver = ReadingTargetResolver()
    chapters = []
    for chapter in document.chapters:
        sections = []
        for section in chapter.sections:
            units = []
            for unit in section.task_units:
                blocks = resolver.content_blocks_for_target_validation(unit,section,chapter)
                updated = []
                for block in blocks:
                    edit = state["blocks"].get(block.block_id)
                    if edit and edit.get("deleted"):
                        continue
                    updated.append(replace(block,content=edit["content"] if edit else block.content))
                content = "\n\n".join(b.content for b in updated)
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                updated = [replace(b,metadata={**(b.metadata or {}),"reading_overlay":True,
                            "source_hash":content_hash,"reading_revision":state.get("revision",0)}) for b in updated]
                units.append(replace(unit,content=content,content_blocks=updated,task_artifacts=None))
            sections.append(replace(section,task_units=units,
                content="\n\n".join(u.content for u in units) if units else section.content,task_artifacts=None))
        chapters.append(replace(chapter,sections=sections,task_artifacts=None))
    return replace(document,chapters=chapters,parse_provenance={**document.parse_provenance,"reading_projected":True})


class ReadingContentService:
    def __init__(self,repository,loader_factory=None):
        self.repository = repository
        self.loader_factory = loader_factory

    def snapshot(self,document):
        state = state_for(document)
        edits = state.get("blocks",{})
        return {"document_id":document.document_id,"revision":state.get("revision",0),
                "source_token":source_token_for(document),"max_chars":MAX_CHARS,
                "disposition":"needs_review" if needs_review(document) else "completed",
                "regions":review_regions(document),"can_undo":bool(state.get("undo")),
                "blocks":[{"block_id":b.block_id,"task_unit_id":u.unit_id,"section_id":s.section_id,
                           "chapter_id":c.chapter_id,"content":edits.get(b.block_id,{}).get("content",b.content),
                           "deleted":edits.get(b.block_id,{}).get("deleted",False),
                           "modified":b.block_id in edits} for c,s,u,b in original_blocks(document)]}

    def read(self,doc_name):
        self.validate_name(doc_name)
        return self.snapshot(self.repository.load_document(doc_name))

    @staticmethod
    def validate_name(doc_name):
        if not doc_name.strip() or doc_name in {".",".."} or any(c in doc_name for c in ("/","\\","\x00")):
            raise ValueError("invalid_document_name")

    def mutate(self,doc_name,*,action,expected_revision,source_token: str,mutation_id,
               block_id=None,text=None,region_id=None):
        self.validate_name(doc_name)
        if action not in {"edit","delete","undo","discard_region"}:
            raise ValueError("unsupported_reading_action")
        if not mutation_id or len(mutation_id)>100 or type(expected_revision) is not int or expected_revision<0:
            raise ValueError("invalid_reading_revision_or_mutation")
        if action=="edit" and (not isinstance(text,str) or not text.strip() or len(text)>MAX_CHARS):
            raise ValueError("reading_text_requires_1_to_10000_characters")
        command = digest([action,expected_revision,source_token,block_id,text,region_id])

        def mutation(document):
            state = deepcopy(state_for(document))
            receipts = state.setdefault("receipts",{})
            if mutation_id in receipts:
                if receipts[mutation_id] != command:
                    raise ReadingContentConflict("reading_mutation_conflict")
                return document,self.snapshot(document)
            if source_token != source_token_for(document) or expected_revision != state.get("revision",0):
                raise ReadingContentConflict("reading_content_changed_reload_required")
            if len(receipts)>=1000:
                raise ReadingContentConflict("reading_history_limit_reached")
            previous = {"blocks":deepcopy(state.get("blocks",{})),"regions":deepcopy(state.get("regions",{}))}
            if action in {"edit","delete"}:
                matches = [b for _,_,_,b in original_blocks(document) if b.block_id==block_id]
                if len(matches)!=1:
                    raise ReadingContentConflict("reading_block_missing")
                prior = state.setdefault("blocks",{}).get(block_id,{})
                if prior.get("deleted"):
                    raise ReadingContentConflict("reading_block_deleted_use_undo")
                state["blocks"][block_id] = {"content":text if action=="edit" else prior.get("content",matches[0].content),
                                             "deleted":action=="delete"}
            elif action=="discard_region":
                if not any(r["id"]==region_id for r in review_regions(document)):
                    raise ReadingContentConflict("review_region_missing")
                state.setdefault("regions",{})[region_id] = True
            else:
                undo = state.get("undo")
                if not undo:
                    raise ReadingContentConflict("nothing_to_undo")
                state.update(undo)
            state["undo"] = None if action=="undo" else previous
            state["revision"] = expected_revision+1
            receipts[mutation_id] = command
            state.setdefault("audit",[]).append({"action":action,"block_id":block_id,"region_id":region_id,
                "revision":state["revision"],"at":datetime.now(timezone.utc).isoformat()})
            artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
            # All generated context depends on ancestors; retain user annotations separately.
            metadata = {k:v for k,v in artifacts.metadata.items() if k not in {
                "reading_interaction_artifacts","ancestor_summary_artifacts","public_summary"}}
            updated = replace(document,parse_provenance={**document.parse_provenance,STATE_KEY:state},
                              document_task_artifacts=replace(artifacts,metadata=metadata,chapter_artifacts={}))
            return updated,self.snapshot(updated)

        return self.repository.mutate_reading_content(doc_name,mutation)

    def preview(self,doc_name,region_id):
        self.validate_name(doc_name)
        document = self.repository.load_document(doc_name)
        review = document.parse_provenance.get("source_review",{})
        region = next((r for r in review.get("regions",[]) if r["id"]==region_id),None)
        if region is None:
            raise FileNotFoundError("review_region_missing")
        loader = self.loader_factory.get(doc_name)
        preview = getattr(loader,"render_source_region",None)
        if not callable(preview):
            raise ReadingContentConflict("source_preview_unavailable")
        return preview(doc_name,review["source_sha256"],region)
