from __future__ import annotations

from paper_ocr.table_cell_ocr_experiment import (
    _apply_cell_text_to_grid,
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
