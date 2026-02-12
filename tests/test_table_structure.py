from __future__ import annotations

import json
from pathlib import Path

import fitz

from paper_ocr.table_structure import build_table_structure_artifacts


def _make_pdf(path: Path) -> None:
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 120), "Table 1 sample")
        doc.save(path)


def _make_doc(tmp_path: Path) -> tuple[Path, str]:
    doc_dir = tmp_path / "out" / "group" / "doc_abc"
    marker_dir = doc_dir / "metadata" / "assets" / "structured" / "marker"
    extracted_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted"
    tables_dir = extracted_dir / "tables"
    marker_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = tmp_path / "source.pdf"
    _make_pdf(pdf_path)
    (doc_dir / "metadata" / "manifest.json").write_text(json.dumps({"source_path": str(pdf_path)}))

    marker_row = {
        "table_group_id": "page_0001_table_01",
        "page": 1,
        "bbox": [72.0, 160.0, 500.0, 360.0],
    }
    (marker_dir / "tables_raw.jsonl").write_text(json.dumps(marker_row) + "\n")

    fragment_row = {
        "fragment_id": "frag-1",
        "table_group_id": "page_0001_table_01",
        "page": 1,
        "header_rows": [["A", "B"]],
        "data_rows": [["1", "2"]],
    }
    (extracted_dir / "table_fragments.jsonl").write_text(json.dumps(fragment_row) + "\n")

    table_id = "p0001_t01"
    manifest_row = {
        "table_id": table_id,
        "page": 1,
        "caption": "Table 1",
        "headers": ["A", "B"],
        "rows": [["1", "2"]],
        "csv_path": f"metadata/assets/structured/extracted/tables/{table_id}.csv",
    }
    (tables_dir / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")
    (tables_dir / f"{table_id}.json").write_text(
        json.dumps(
            {
                "table_id": table_id,
                "pages": [1],
                "fragment_ids": ["frag-1"],
                "header_rows_full": [["A", "B"]],
                "data_rows": [["1", "2"]],
            }
        )
    )
    return doc_dir, table_id


def test_build_table_structure_artifacts_generates_structure(monkeypatch, tmp_path: Path):
    doc_dir, table_id = _make_doc(tmp_path)

    def _fake_run(command, shell, check, stdout, stderr, timeout, text):  # noqa: ANN001
        rendered = str(command)
        marker = "--output "
        start = rendered.find(marker)
        assert start >= 0
        output_path = Path(rendered[start + len(marker) :].strip().strip('"'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "rows": 2,
                    "cols": 2,
                    "header_rows": 1,
                    "cells": [
                        {"row_start": 0, "row_end": 0, "col_start": 0, "col_end": 0},
                        {"row_start": 0, "row_end": 0, "col_start": 1, "col_end": 1},
                        {"row_start": 1, "row_end": 1, "col_start": 0, "col_end": 0},
                        {"row_start": 1, "row_end": 1, "col_start": 1, "col_end": 1},
                    ],
                }
            )
        )

        class _Result:
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("paper_ocr.table_structure.subprocess.run", _fake_run)
    summary = build_table_structure_artifacts(
        doc_dir=doc_dir,
        model="tatr",
        command='mock_tatr --image "{image}" --output "{output}"',
        timeout=10,
    )
    assert summary["enabled"] is True
    assert summary["generated"] == 1
    structure_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_structure" / f"{table_id}.json"
    payload = json.loads(structure_path.read_text())
    assert payload["rows"] == 2
    assert payload["cols"] == 2
    manifest_rows = [
        json.loads(line)
        for line in (doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_structure" / "manifest.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert manifest_rows[0]["table_id"] == table_id
    assert manifest_rows[0]["status"] == "ok"


def test_build_table_structure_artifacts_requires_command(tmp_path: Path):
    doc_dir, _ = _make_doc(tmp_path)
    summary = build_table_structure_artifacts(
        doc_dir=doc_dir,
        model="tatr",
        command="",
        timeout=10,
    )
    assert summary["enabled"] is False
    assert summary["command_configured"] is False
    assert "table_structure_command_missing" in summary["errors"]
