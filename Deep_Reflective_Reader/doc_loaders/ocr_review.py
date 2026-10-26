"""Format-neutral, source-linked review evidence; not parser authority."""

import hashlib

REVIEW_POLICY = "source-review-v2-preserve-text"


def region_record(page, index, bbox, dpi, reason, candidate="", *, excluded_from_body=False):
    return {"id":f"page-{page}-region-{index}","page_index":page,"bbox":list(bbox),
            "render_dpi":dpi,"reason":reason,"raw_candidate":candidate,
            **({"body_inclusion": "excluded"} if excluded_from_body else {}),
            "status":"unresolved"}


def review_record(source_hash, regions):
    return {"policy":REVIEW_POLICY,"source_sha256":source_hash,
            "disposition":"needs_review" if regions else "completed","regions":regions}


def page_review_regions(page_index, text, summary, bbox, dpi):
    """Describe uncertainty without changing the selected page transcription."""
    records = []
    regions = summary.get("regions", [])
    next_component = len(regions)
    for index, region in enumerate(regions):
        if region.get("accepted") is False or region.get("quality_flags"):
            start, end = region.get("char_start"), region.get("char_end")
            mapped = type(start) is int and type(end) is int and 0 <= start < end <= len(text)
            candidate = text[start:end] if mapped else region.get("raw_candidate") or ""
            record = region_record(page_index, index, region["bbox"], dpi,
                                   "recognition_uncertain" if candidate.strip() else "recognition_empty",
                                   candidate)
            if mapped:
                record.update(body_inclusion="included", page_span={"char_start": start, "char_end": end})
            records.append(record)
        # A retained parent column must not conceal a missing title/digit component.
        for component in region.get("components", []):
            start, end = component.get("char_start"), component.get("char_end")
            if (type(start) is int and type(end) is int and 0 <= start == end <= len(text)
                    and component.get("bbox")):
                records.append(region_record(page_index, next_component, component["bbox"], dpi,
                                             "recognition_empty", excluded_from_body=True))
                next_component += 1
    if not records and (not text.strip() or not summary.get("selected_passed")):
        record = region_record(page_index, 0, bbox, dpi,
                               "recognition_uncertain" if text.strip() else "recognition_empty", text)
        if text.strip():
            record.update(body_inclusion="included", page_span={"char_start": 0, "char_end": len(text)})
        else:
            record["body_inclusion"] = "excluded"
        records.append(record)
    return records


def bind_review_spans(regions, text, boundaries):
    """Translate loader-owned page offsets only after final canonical assembly."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    pages = {page.page_index: page for page in boundaries}
    bound = []
    for source in regions:
        record = dict(source)
        span = record.pop("page_span", None)
        page = pages.get(record.get("page_index"))
        if span is not None and page is not None:
            start, end = span["char_start"], span["char_end"]
            candidate = record.get("raw_candidate")
            if (0 <= start < end <= len(page.text) and page.text[start:end] == candidate
                    and text[page.char_start + start:page.char_start + end] == candidate):
                record["canonical_span"] = {
                    "char_start": page.char_start + start, "char_end": page.char_start + end,
                    "text_sha256": digest,
                }
        bound.append(record)
    return bound
