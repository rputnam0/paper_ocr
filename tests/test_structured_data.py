from __future__ import annotations

import json
from pathlib import Path

from paper_ocr.structured_data import build_structured_exports, extract_markdown_tables


def test_extract_markdown_tables_parses_header_and_rows():
    md = """
Intro text.

Table 1: Properties
| Polymer | Mw |
| --- | --- |
| A | 12000 |
| B | 32000 |
"""
    tables = extract_markdown_tables(md)
    assert len(tables) == 1
    assert tables[0]["caption"] == "Table 1: Properties"
    assert tables[0]["headers"] == ["Polymer", "Mw"]
    assert tables[0]["rows"] == [["A", "12000"], ["B", "32000"]]


def test_extract_markdown_tables_keeps_empty_edge_cells():
    md = """
| | A | B | |
| --- | --- | --- | --- |
| | 1 | 2 | |
"""
    tables = extract_markdown_tables(md)
    assert len(tables) == 1
    assert tables[0]["headers"] == ["", "A", "B", ""]
    assert tables[0]["rows"] == [["", "1", "2", ""]]


def test_build_structured_exports_writes_tables_and_figures(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    marker_assets = doc_dir / "metadata" / "assets" / "structured" / "marker" / "page_0001_assets" / "page_0001"
    marker_assets.mkdir(parents=True)
    (marker_assets / "_page_0_Figure_1.jpeg").write_bytes(b"jpg")

    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "Table 1: Data",
                "| Name | Value |",
                "| --- | --- |",
                "| X | 1 |",
                "",
                "![](_page_0_Figure_1.jpeg)",
                "",
            ]
        )
    )

    summary = build_structured_exports(doc_dir)

    assert summary.table_count == 1
    assert summary.figure_count == 1
    assert summary.deplot_count == 0
    assert (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.csv").exists()
    assert (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "figures" / "manifest.jsonl").exists()
    manifest = json.loads((doc_dir / "metadata" / "assets" / "structured" / "extracted" / "manifest.json").read_text())
    assert manifest["table_count"] == 1
    assert manifest["figure_count"] == 1
    assert manifest["tables_manifest"] == "metadata/assets/structured/extracted/tables/manifest.jsonl"
    assert manifest["figures_manifest"] == "metadata/assets/structured/extracted/figures/manifest.jsonl"

    table_payload = json.loads(
        (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json").read_text()
    )
    assert table_payload["csv_path"] == "metadata/assets/structured/extracted/tables/p0001_t01.csv"
    figure_manifest_line = (
        doc_dir / "metadata" / "assets" / "structured" / "extracted" / "figures" / "manifest.jsonl"
    ).read_text().strip()
    figure_payload = json.loads(figure_manifest_line)
    assert figure_payload["resolved_path"] == (
        "metadata/assets/structured/marker/page_0001_assets/page_0001/_page_0_Figure_1.jpeg"
    )


def test_build_structured_exports_runs_deplot_command(monkeypatch, tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    marker_assets = doc_dir / "metadata" / "assets" / "structured" / "marker" / "page_0001_assets" / "page_0001"
    marker_assets.mkdir(parents=True)
    image_path = marker_assets / "_page_0_Figure_1.jpeg"
    image_path.write_bytes(b"jpg")

    (pages_dir / "0001.md").write_text("![](_page_0_Figure_1.jpeg)\n")

    def _fake_run(cmd, check, stdout, stderr, text, timeout):  # noqa: ANN001
        assert str(image_path) in cmd
        class _Proc:
            returncode = 0
            stdout = '{"series":[{"x":[1,2],"y":[3,4]}]}'
            stderr = ""

        return _Proc()

    monkeypatch.setattr("paper_ocr.structured_data.subprocess.run", _fake_run)

    summary = build_structured_exports(
        doc_dir=doc_dir,
        deplot_command="deplot-cli --image {image}",
        deplot_timeout=30,
    )
    assert summary.figure_count == 1
    assert summary.deplot_count == 1
    out_json = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "figures" / "deplot" / "p0001_f01.json"
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert payload["output"]["series"][0]["x"] == [1, 2]


def test_build_structured_exports_cleans_stale_artifacts_on_rerun(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)

    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "| Name | Value |",
                "| --- | --- |",
                "| X | 1 |",
                "",
            ]
        )
    )
    first = build_structured_exports(doc_dir)
    assert first.table_count == 1
    stale_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "stale.csv"
    stale_path.write_text("stale")
    assert stale_path.exists()

    # Rerun with no tables; old table and stale files should be removed.
    (pages_dir / "0001.md").write_text("No tables now.\n")
    second = build_structured_exports(doc_dir)
    assert second.table_count == 0
    assert not stale_path.exists()
    table_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    assert not (table_dir / "p0001_t01.csv").exists()


def test_build_structured_exports_prefers_marker_tables_raw(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "| Wrong | Table |",
                "| --- | --- |",
                "| X | 1 |",
            ]
        )
    )
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_table = {
        "table_group_id": "tblgrp-1",
        "table_block_ids": ["b1", "b2"],
        "caption_block_id": "c1",
        "page": 1,
        "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
        "header_rows": [["Polymer", "Value"]],
        "data_rows": [["A", "10"], ["B", "12"]],
        "caption_text": "Table 1: Marker data",
    }
    (marker_root / "tables_raw.jsonl").write_text(json.dumps(marker_table) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 1
    canonical = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "canonical.jsonl"
    rows = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
    assert rows[0]["caption_text"] == "Table 1: Marker data"
    assert rows[0]["source_format"] == "html"


def test_build_structured_exports_writes_qa_flags_for_grobid_disagreement(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text("No markdown tables\n")
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_table = {
        "table_group_id": "tblgrp-1",
        "table_block_ids": ["b1"],
        "caption_block_id": "c1",
        "page": 1,
        "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
        "header_rows": [["Polymer", "Value"]],
        "data_rows": [["A", "10"]],
        "caption_text": "Table 1",
    }
    (marker_root / "tables_raw.jsonl").write_text(json.dumps(marker_table) + "\n")
    grobid_root = doc_dir / "metadata" / "assets" / "structured" / "grobid"
    grobid_root.mkdir(parents=True, exist_ok=True)
    grobid_rec = {"doc_id": "doc", "type": "table", "label": "Table 2", "page": 3, "coords": []}
    (grobid_root / "figures_tables.jsonl").write_text(json.dumps(grobid_rec) + "\n")

    summary = build_structured_exports(
        doc_dir=doc_dir,
        table_source="marker-first",
        table_qa_mode="warn",
    )
    assert summary.table_count == 1
    qa_flags = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_flags.jsonl"
    flags = [json.loads(line) for line in qa_flags.read_text().splitlines() if line.strip()]
    assert any(flag["type"] in {"count_mismatch", "page_mismatch", "caption_number_mismatch"} for flag in flags)
