from __future__ import annotations

from paper_ocr.table_cell_ocr_experiment import (
    _apply_cell_text_to_grid,
    _build_llm_reconciliation_prompt,
    _normalize_llm_reconciliation_payload,
    _normalize_structure_payload_with_bboxes,
)


def test_normalize_structure_payload_with_bboxes_supports_normalized_coordinates():
    payload = {
        "row_count": 2,
        "col_count": 2,
        "cells": [
            {"row": 0, "col": 0, "bbox": [0.0, 0.0, 0.5, 0.5]},
            {"row": 0, "col": 1, "bbox": [0.5, 0.0, 1.0, 0.5]},
            {"row": 1, "col": 0, "bbox": [0.0, 0.5, 0.5, 1.0]},
            {"row": 1, "col": 1, "bbox": [0.5, 0.5, 1.0, 1.0]},
        ],
    }

    normalized = _normalize_structure_payload_with_bboxes(payload, crop_width=400, crop_height=200, bbox_mode="auto")

    assert normalized["rows"] == 2
    assert normalized["cols"] == 2
    assert len(normalized["cells"]) == 4
    assert normalized["cells"][0]["bbox"] == [0, 0, 200, 100]
    assert normalized["cells"][3]["bbox"] == [200, 100, 400, 200]


def test_normalize_structure_payload_with_bboxes_infers_grid_and_supports_pixel_mode():
    payload = {
        "cells": [
            {
                "row_start": 0,
                "row_end": 0,
                "col_start": 0,
                "col_end": 1,
                "cell_bbox": [0, 0, 320, 100],
            },
            {
                "row_start": 1,
                "row_end": 1,
                "col_start": 0,
                "col_end": 0,
                "cell_bbox": [0, 100, 160, 200],
            },
            {
                "row_start": 1,
                "row_end": 1,
                "col_start": 1,
                "col_end": 1,
                "cell_bbox": [160, 100, 320, 200],
            },
        ]
    }

    normalized = _normalize_structure_payload_with_bboxes(payload, crop_width=320, crop_height=200, bbox_mode="pixels")

    assert normalized["rows"] == 2
    assert normalized["cols"] == 2
    assert normalized["cells"][0]["bbox"] == [0, 0, 320, 100]
    assert normalized["cells"][0]["col_end"] == 1


def test_apply_cell_text_to_grid_assigns_spans():
    grid = _apply_cell_text_to_grid(
        rows=2,
        cols=2,
        cells=[
            {"row_start": 0, "row_end": 0, "col_start": 0, "col_end": 1, "text": "Header"},
            {"row_start": 1, "row_end": 1, "col_start": 0, "col_end": 0, "text": "A"},
            {"row_start": 1, "row_end": 1, "col_start": 1, "col_end": 1, "text": "B"},
        ],
    )

    assert grid == [["Header", "Header"], ["A", "B"]]


def test_normalize_structure_payload_derives_cell_bboxes_from_row_column_detections():
    payload = {
        "rows": 2,
        "cols": 2,
        "cells": [
            {"row_start": 0, "row_end": 0, "col_start": 0, "col_end": 0},
            {"row_start": 0, "row_end": 0, "col_start": 1, "col_end": 1},
            {"row_start": 1, "row_end": 1, "col_start": 0, "col_end": 0},
            {"row_start": 1, "row_end": 1, "col_start": 1, "col_end": 1},
        ],
        "detections": [
            {"label": "table row", "box": [0, 0, 200, 50]},
            {"label": "table row", "box": [0, 50, 200, 100]},
            {"label": "table column", "box": [0, 0, 100, 100]},
            {"label": "table column", "box": [100, 0, 200, 100]},
        ],
    }

    normalized = _normalize_structure_payload_with_bboxes(payload, crop_width=200, crop_height=100, bbox_mode="pixels")

    assert normalized["rows"] == 2
    assert normalized["cols"] == 2
    assert len(normalized["cells"]) == 4
    assert normalized["cells"][0]["bbox"] == [0, 0, 100, 50]
    assert normalized["cells"][3]["bbox"] == [100, 50, 200, 100]


def test_build_llm_reconciliation_prompt_includes_required_inputs():
    prompt = _build_llm_reconciliation_prompt(
        table_id="t1",
        structure={"rows": 2, "cols": 2, "header_rows": 1, "cells": []},
        prefilled_grid=[["A", "B"], ["1", "2"]],
        full_table_text_lines=["A | B", "1 | 2"],
        join_hints=[{"cells": [[0, 0], [0, 1]], "text": "A B"}],
        row_kinds=["header", "data"],
        cell_level_debug={"cell_level_ocr_debug_only": True, "authoritative_for_text": False, "cell_count": 2},
    )

    assert "table_structure" in prompt
    assert "prefilled_grid_from_cell_ocr" in prompt
    assert "full_table_text_lines" in prompt
    assert "join_hints" in prompt
    assert "row_kinds" in prompt
    assert "cell_level_ocr_debug_only" in prompt
    assert "corrected_header_rows_full" in prompt
    assert "corrected_rows" in prompt


def test_normalize_llm_reconciliation_payload_enforces_shape_and_normalizes_empty():
    normalized = _normalize_llm_reconciliation_payload(
        raw_payload={
            "corrected_header_rows_full": [["H1", "H2"]],
            "corrected_rows": [["1", "<empty>"], ["3", "4"]],
            "applied_corrections": True,
            "notes": "fixed trailing cell",
        },
        expected_rows=2,
        expected_cols=2,
        expected_header_rows=1,
        fallback_grid=[["H1", "H2"], ["1", ""], ["3", "4"]],
    )

    assert normalized["valid"] is True
    assert normalized["header_rows_full"] == [["H1", "H2"]]
    assert normalized["rows"] == [["1", ""], ["3", "4"]]
    assert normalized["final_grid"] == [["H1", "H2"], ["1", ""], ["3", "4"]]
    assert normalized["applied_corrections"] is True


def test_normalize_llm_reconciliation_payload_rejects_drop_and_falls_back():
    normalized = _normalize_llm_reconciliation_payload(
        raw_payload={
            "corrected_header_rows_full": [["H1", "H2"]],
            "corrected_rows": [["1", ""], ["3", "4"]],
            "applied_corrections": True,
            "notes": "dropped token",
        },
        expected_rows=2,
        expected_cols=2,
        expected_header_rows=1,
        fallback_grid=[["H1", "H2"], ["1", "X"], ["3", "4"]],
    )

    assert normalized["valid"] is False
    assert normalized["reason"] == "no_drop_violation"
    assert normalized["rows"] == [["1", "X"], ["3", "4"]]
