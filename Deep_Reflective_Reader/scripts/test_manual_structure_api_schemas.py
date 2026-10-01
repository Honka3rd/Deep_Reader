#!/usr/bin/env python3
"""Schema regression tests for source-agnostic manual structure requests."""

from __future__ import annotations

from pathlib import Path
import sys

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_schemas import (  # noqa: E402
    ManualStructureAnchorRequest,
    ManualStructureValidationResponse,
    ManualStructureValidationRequest,
    ReparseDocumentStructureRequest,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ManualStructureValidationRequest.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in validation error: {error}",
        )
        return
    raise AssertionError(f"expected validation error containing {expected_message!r}")


def _assert_response_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ManualStructureValidationResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in response validation error: {error}",
        )
        return
    raise AssertionError(f"expected response validation error containing {expected_message!r}")


def _assert_reparse_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ReparseDocumentStructureRequest.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in reparse validation error: {error}",
        )
        return
    raise AssertionError(f"expected reparse validation error containing {expected_message!r}")


def test_manual_structure_accepts_char_range_plan() -> None:
    request = ManualStructureValidationRequest.model_validate(
        {
            "doc_name": "  Any Source.txt  ",
            "manual_structure": {
                "source_hash": "  raw-hash  ",
                "entries": [
                    {
                        "title": "  Chapter One  ",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char-range",
                            "char_start": 0,
                            "char_end": 100,
                        },
                        "external_id": "  ch-1  ",
                        "notes": "  reviewed  ",
                    },
                    {
                        "title": "Section One",
                        "level": 2,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 20,
                            "char_end": 80,
                        },
                    },
                ],
            },
        }
    )

    _assert(request.doc_name == "Any Source.txt", "doc_name should be trimmed")
    _assert(request.manual_structure.source_hash == "raw-hash", "source hash should be trimmed")
    first_entry = request.manual_structure.entries[0]
    _assert(first_entry.title == "Chapter One", "entry title should be trimmed")
    _assert(first_entry.external_id == "ch-1", "external_id should be trimmed")
    _assert(first_entry.notes == "reviewed", "notes should be trimmed")
    _assert(first_entry.anchor.anchor_type == "char_range", "anchor type should normalize")


def test_manual_structure_accepts_page_range_plan() -> None:
    request = ManualStructureValidationRequest.model_validate(
        {
            "doc_name": "Book.pdf",
            "manual_structure": {
                "entries": [
                    {
                        "title": "Chapter One",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "page_range",
                            "page_start_index": 3,
                            "page_end_index": 8,
                        },
                    }
                ],
            },
        }
    )

    anchor = request.manual_structure.entries[0].anchor
    _assert(anchor.page_start_index == 3, "page_start_index should be preserved")
    _assert(anchor.page_end_index == 8, "page_end_index should be preserved")


def test_manual_structure_rejects_invalid_shapes() -> None:
    base_payload = {
        "doc_name": "Book",
        "manual_structure": {
            "entries": [
                {
                    "title": "Chapter One",
                    "level": 1,
                    "anchor": {
                        "anchor_type": "char_range",
                        "char_start": 0,
                    },
                }
            ],
        },
    }

    empty_title_payload = dict(base_payload)
    empty_title_payload["manual_structure"] = {
        "entries": [
            {
                "title": "   ",
                "level": 1,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
            }
        ]
    }
    _assert_validation_error(empty_title_payload, "manual structure entry title cannot be empty")

    too_deep_payload = dict(base_payload)
    too_deep_payload["manual_structure"] = {
        "entries": [
            {
                "title": "Deep Child",
                "level": 3,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
            }
        ]
    }
    _assert_validation_error(too_deep_payload, "less than or equal to 2")

    orphan_section_payload = dict(base_payload)
    orphan_section_payload["manual_structure"] = {
        "entries": [
            {
                "title": "Orphan Section",
                "level": 2,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
            }
        ]
    }
    _assert_validation_error(
        orphan_section_payload,
        "manual structure section entry requires preceding chapter",
    )

    mixed_anchor_payload = dict(base_payload)
    mixed_anchor_payload["manual_structure"] = {
        "entries": [
            {
                "title": "Chapter One",
                "level": 1,
                "anchor": {
                    "anchor_type": "char_range",
                    "char_start": 0,
                    "page_start_index": 1,
                },
            }
        ]
    }
    _assert_validation_error(
        mixed_anchor_payload,
        "manual structure anchor must not mix char and page fields",
    )


def test_manual_structure_anchor_rejects_bad_ranges() -> None:
    try:
        ManualStructureAnchorRequest.model_validate(
            {
                "anchor_type": "char_range",
                "char_start": 10,
                "char_end": 10,
            }
        )
    except ValidationError as error:
        _assert("char_end must be greater than char_start" in str(error), "expected char range error")
    else:
        raise AssertionError("expected char range validation error")

    try:
        ManualStructureAnchorRequest.model_validate(
            {
                "anchor_type": "page_range",
                "page_start_index": 10,
                "page_end_index": 9,
            }
        )
    except ValidationError as error:
        _assert(
            "page_end_index must be greater than or equal to page_start_index" in str(error),
            "expected page range error",
        )
    else:
        raise AssertionError("expected page range validation error")


def test_manual_structure_validation_response_accepts_preview_payload() -> None:
    response = ManualStructureValidationResponse.model_validate(
        {
            "doc_name": "  Book.pdf  ",
            "valid": True,
            "normalized_entries": [
                {
                    "title": "  Chapter One  ",
                    "level": 1,
                    "anchor": {"anchor_type": "char-range", "char_start": 0, "char_end": 100},
                    "external_id": "  ch-1  ",
                    "projected_char_start": 0,
                    "projected_char_end": 100,
                },
                {
                    "title": "Section One",
                    "level": 2,
                    "anchor": {"anchor_type": "page_range", "page_start_index": 2},
                    "projected_page_start_index": 2,
                    "projected_page_end_index": 3,
                },
            ],
            "warnings": [
                {
                    "code": "stale-source-evidence",
                    "message": " Source hash differs from current source. ",
                    "severity": "WARNING",
                    "entry_index": 0,
                    "external_id": " ch-1 ",
                }
            ],
            "preview_chapters": [
                {
                    "title": " Chapter One ",
                    "external_id": " ch-1 ",
                    "entry_index": 0,
                    "sections": [
                        {
                            "title": " Section One ",
                            "entry_index": 1,
                        }
                    ],
                }
            ],
            "parse_provenance_preview": {
                "parser_mode": "manual-structure",
                "source_hash": " raw-hash ",
                "anchor_types": ["char-range", "page_range"],
            },
        }
    )

    _assert(response.doc_name == "Book.pdf", "response doc_name should be trimmed")
    _assert(response.valid is True, "response should preserve valid flag")
    first_entry = response.normalized_entries[0]
    _assert(first_entry.title == "Chapter One", "normalized entry title should be trimmed")
    _assert(first_entry.anchor.anchor_type == "char_range", "entry anchor type should normalize")
    _assert(first_entry.external_id == "ch-1", "entry external_id should be trimmed")
    warning = response.warnings[0]
    _assert(warning.code == "stale_source_evidence", "issue code should normalize")
    _assert(warning.severity == "warning", "issue severity should normalize")
    provenance = response.parse_provenance_preview
    _assert(provenance is not None, "provenance preview should be present")
    _assert(provenance.parser_mode == "manual_structure", "parser mode should normalize")
    _assert(
        provenance.anchor_types == ["char_range", "page_range"],
        "anchor types should normalize",
    )


def test_manual_structure_validation_response_rejects_invalid_payloads() -> None:
    valid_base = {
        "doc_name": "Book.pdf",
        "valid": True,
        "normalized_entries": [
            {
                "title": "Chapter One",
                "level": 1,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
            }
        ],
    }

    response_with_error = dict(valid_base)
    response_with_error["errors"] = [
        {
            "code": "malformed_payload",
            "message": "bad payload",
            "severity": "error",
        }
    ]
    _assert_response_validation_error(
        response_with_error,
        "manual structure validation response cannot be valid with errors",
    )

    unsupported_code = {
        "doc_name": "Book.pdf",
        "valid": False,
        "errors": [
            {
                "code": "unknown_code",
                "message": "bad payload",
                "severity": "error",
            }
        ],
    }
    _assert_response_validation_error(
        unsupported_code,
        "unsupported manual structure validation issue code",
    )

    unsupported_severity = {
        "doc_name": "Book.pdf",
        "valid": False,
        "errors": [
            {
                "code": "malformed_payload",
                "message": "bad payload",
                "severity": "fatal",
            }
        ],
    }
    _assert_response_validation_error(
        unsupported_severity,
        "manual structure validation issue severity must be error, warning, or info",
    )

    unsupported_depth = {
        "doc_name": "Book.pdf",
        "valid": True,
        "normalized_entries": [
            {
                "title": "Subsection",
                "level": 3,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
            }
        ],
    }
    _assert_response_validation_error(unsupported_depth, "less than or equal to 2")

    bad_range = {
        "doc_name": "Book.pdf",
        "valid": True,
        "normalized_entries": [
            {
                "title": "Chapter One",
                "level": 1,
                "anchor": {"anchor_type": "char_range", "char_start": 0},
                "projected_char_start": 10,
                "projected_char_end": 10,
            }
        ],
    }
    _assert_response_validation_error(
        bad_range,
        "projected_char_end must be greater than projected_char_start",
    )

    bad_parser_mode = {
        "doc_name": "Book.pdf",
        "valid": True,
        "parse_provenance_preview": {
            "parser_mode": "llm_enhanced",
            "anchor_types": ["char_range"],
        },
    }
    _assert_response_validation_error(
        bad_parser_mode,
        "manual structure preview parser_mode must be manual_structure",
    )


def test_manual_structure_reparse_request_accepts_manual_structure_plan() -> None:
    request = ReparseDocumentStructureRequest.model_validate(
        {
            "doc_name": "  Book.pdf  ",
            "parser_mode": "manual-structure",
            "manual_structure": {
                "source_hash": " source-hash ",
                "entries": [
                    {
                        "title": "Chapter One",
                        "level": 1,
                        "anchor": {
                            "anchor_type": "char_range",
                            "char_start": 0,
                            "char_end": 100,
                        },
                    },
                    {
                        "title": "Section One",
                        "level": 2,
                        "anchor": {
                            "anchor_type": "page-range",
                            "page_start_index": 2,
                        },
                    },
                ],
            },
        }
    )

    _assert(request.doc_name == "Book.pdf", "reparse doc_name should be trimmed")
    _assert(
        request.parser_mode == "manual_structure",
        "reparse parser_mode should normalize",
    )
    _assert(
        request.manual_structure is not None,
        "manual reparse should preserve manual structure plan",
    )
    _assert(
        request.manual_structure.source_hash == "source-hash",
        "manual reparse source hash should be trimmed",
    )
    _assert(
        request.manual_structure.entries[1].anchor.anchor_type == "page_range",
        "manual reparse entry anchor should normalize",
    )


def test_manual_structure_reparse_request_preserves_existing_modes() -> None:
    common_request = ReparseDocumentStructureRequest.model_validate(
        {
            "doc_name": "Book.pdf",
            "parser_mode": "common",
        }
    )
    llm_request = ReparseDocumentStructureRequest.model_validate(
        {
            "doc_name": "Book.pdf",
            "parser_mode": "llm-enhanced",
        }
    )

    _assert(common_request.parser_mode == "common", "common parser mode should remain valid")
    _assert(
        llm_request.parser_mode == "llm_enhanced",
        "llm_enhanced parser mode should normalize",
    )


def test_manual_structure_reparse_request_rejects_invalid_shapes() -> None:
    manual_without_plan = {
        "doc_name": "Book.pdf",
        "parser_mode": "manual_structure",
    }
    _assert_reparse_validation_error(
        manual_without_plan,
        "manual_structure is required when parser_mode is manual_structure",
    )

    common_with_plan = {
        "doc_name": "Book.pdf",
        "parser_mode": "common",
        "manual_structure": {
            "entries": [
                {
                    "title": "Chapter One",
                    "level": 1,
                    "anchor": {"anchor_type": "char_range", "char_start": 0},
                }
            ],
        },
    }
    _assert_reparse_validation_error(
        common_with_plan,
        "manual_structure is only allowed when parser_mode is manual_structure",
    )

    bad_parser_mode = {
        "doc_name": "Book.pdf",
        "parser_mode": "ocr_toc",
    }
    _assert_reparse_validation_error(
        bad_parser_mode,
        "parser_mode must be common, llm_enhanced, or manual_structure",
    )

    empty_doc_name = {
        "doc_name": "   ",
        "parser_mode": "common",
    }
    _assert_reparse_validation_error(empty_doc_name, "doc_name cannot be empty")


if __name__ == "__main__":
    test_manual_structure_accepts_char_range_plan()
    test_manual_structure_accepts_page_range_plan()
    test_manual_structure_rejects_invalid_shapes()
    test_manual_structure_anchor_rejects_bad_ranges()
    test_manual_structure_validation_response_accepts_preview_payload()
    test_manual_structure_validation_response_rejects_invalid_payloads()
    test_manual_structure_reparse_request_accepts_manual_structure_plan()
    test_manual_structure_reparse_request_preserves_existing_modes()
    test_manual_structure_reparse_request_rejects_invalid_shapes()
    print("OK: manual structure API schema tests passed")
