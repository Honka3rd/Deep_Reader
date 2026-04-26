#!/usr/bin/env python3
"""Minimal REST test for structured parser mode comparison (common vs llm_enhanced)."""

import argparse
import difflib
import json
from pathlib import Path
from typing import Any
from urllib import error, request


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                raise RuntimeError(f"unexpected response payload type: {type(parsed).__name__}")
            return status, parsed
    except error.HTTPError as http_error:
        raw_error = http_error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {http_error.code} {http_error.reason}: {raw_error}"
        ) from http_error
    except error.URLError as url_error:
        raise RuntimeError(f"request failed: {url_error}") from url_error


def _load_json(path: str) -> dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"structured artifact is not a JSON object: {path}")
    return decoded


def _extract_section_titles(document: dict[str, Any]) -> list[str]:
    section_payloads = document.get("sections", [])
    if not isinstance(section_payloads, list):
        return []
    titles: list[str] = []
    for section in section_payloads:
        if not isinstance(section, dict):
            continue
        title = (section.get("title") or "").strip()
        if title:
            titles.append(title)
        else:
            titles.append("<untitled>")
    return titles


def _summarize_document(document: dict[str, Any]) -> dict[str, Any]:
    titles = _extract_section_titles(document)
    return {
        "section_count": len(document.get("sections", []))
        if isinstance(document.get("sections", []), list)
        else 0,
        "first_titles": titles[:12],
    }


def _write_snapshot(path: str, content: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare structured artifact output between common and llm_enhanced parser modes via REST."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--doc-name", required=True, help="Document name used by /documents/prepare")
    parser.add_argument(
        "--mode",
        default="base",
        choices=("base", "free_qa"),
        help="Preparation mode sent to /documents/prepare",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout seconds for each request",
    )
    parser.add_argument(
        "--write-snapshots",
        action="store_true",
        help="Write parser outputs into *.common.snapshot.json and *.llm_enhanced.snapshot.json",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/documents/prepare"
    base_payload = {
        "doc_name": args.doc_name,
        "mode": args.mode,
        "force_rebuild": True,
    }

    common_payload = {**base_payload, "structured_parser_mode": "common"}
    common_status, common_response = _post_json(endpoint, common_payload, args.timeout)
    common_path = common_response.get("structured_document_path")
    if not isinstance(common_path, str) or not common_path.strip():
        raise RuntimeError(f"common mode returned invalid structured_document_path: {common_response}")
    common_document = _load_json(common_path)

    llm_payload = {**base_payload, "structured_parser_mode": "llm_enhanced"}
    llm_status, llm_response = _post_json(endpoint, llm_payload, args.timeout)
    llm_path = llm_response.get("structured_document_path")
    if not isinstance(llm_path, str) or not llm_path.strip():
        raise RuntimeError(f"llm_enhanced mode returned invalid structured_document_path: {llm_response}")
    llm_document = _load_json(llm_path)

    common_titles = _extract_section_titles(common_document)
    llm_titles = _extract_section_titles(llm_document)
    diff_lines = list(
        difflib.unified_diff(
            common_titles,
            llm_titles,
            fromfile="common_titles",
            tofile="llm_enhanced_titles",
            lineterm="",
        )
    )

    snapshot_paths: dict[str, str] = {}
    if args.write_snapshots:
        snapshot_paths["common"] = _write_snapshot(
            f"{common_path}.common.snapshot.json",
            common_document,
        )
        snapshot_paths["llm_enhanced"] = _write_snapshot(
            f"{llm_path}.llm_enhanced.snapshot.json",
            llm_document,
        )

    report = {
        "doc_name": args.doc_name,
        "mode": args.mode,
        "requests": {
            "common": {"status": common_status, "payload": common_payload, "response": common_response},
            "llm_enhanced": {
                "status": llm_status,
                "payload": llm_payload,
                "response": llm_response,
            },
        },
        "single_source_path": {
            "common_path": common_path,
            "llm_enhanced_path": llm_path,
            "same_path": common_path == llm_path,
        },
        "summary": {
            "common": _summarize_document(common_document),
            "llm_enhanced": _summarize_document(llm_document),
            "titles_added_in_llm": sorted(set(llm_titles) - set(common_titles))[:20],
            "titles_removed_in_llm": sorted(set(common_titles) - set(llm_titles))[:20],
            "title_diff_preview": diff_lines[:80],
        },
        "snapshots": snapshot_paths,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
