"""Format-neutral, source-linked review evidence; not parser authority."""

import os

REVIEW_POLICY = "source-review-v1"


def enabled():
    return os.environ.get("DEEP_READER_OCR_REVIEW_ENABLED", "1") == "1"


def region_record(page, index, bbox, dpi, reason, candidate=""):
    return {"id":f"page-{page}-region-{index}","page_index":page,"bbox":list(bbox),
            "render_dpi":dpi,"reason":reason,"raw_candidate":candidate,
            "status":"unresolved"}


def review_record(source_hash, regions):
    return {"policy":REVIEW_POLICY,"source_sha256":source_hash,
            "disposition":"needs_review" if regions else "completed","regions":regions}
