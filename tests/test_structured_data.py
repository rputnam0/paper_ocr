from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_ocr.structured_data import (
    build_structured_exports,
    compare_marker_tables_with_ocr_html,
    extract_markdown_tables,
)


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


def test_extract_markdown_tables_normalizes_line_break_cells_in_csv(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "Table 1: Data",
                "| Name | Note |",
                "| --- | --- |",
                "| A | first<br>line |",
            ]
        )
    )

    summary = build_structured_exports(doc_dir)
    assert summary.table_count == 1
    csv_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.csv"
    text = csv_path.read_text()
    assert "<br>" not in text
    assert "first line" in text


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


def test_build_structured_exports_parses_html_when_marker_row_has_no_grid(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_table = {
        "table_group_id": "tblgrp-1",
        "table_block_ids": ["b1"],
        "caption_block_id": "c1",
        "page": 1,
        "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
        "caption_text": "Table 1: HTML-only",
        "html_table": (
            "<table><thead><tr><th>Condition</th><th>\u03c4 (Pa)</th></tr></thead>"
            "<tbody><tr><td>A</td><td>12.0 \u00b1 0.2</td></tr></tbody></table>"
        ),
    }
    (marker_root / "tables_raw.jsonl").write_text(json.dumps(marker_table) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 1
    table_manifest = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    first_row = json.loads(table_manifest.read_text().splitlines()[0])
    csv_path = doc_dir / first_row["csv_path"]
    csv_text = csv_path.read_text()
    assert "Condition,\u03c4 (Pa)" in csv_text
    assert "A,12.0 \u00b1 0.2" in csv_text


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


def test_build_structured_exports_does_not_merge_distant_same_number_tables(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "page": 1,
            "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
            "header_rows": [["Polymer", "Value"]],
            "data_rows": [["A", "10"]],
            "caption_text": "Table 1: Main text",
        },
        {
            "page": 8,
            "polygons": [[[15, 15], [120, 15], [120, 210], [15, 210]]],
            "header_rows": [["Sample", "Result"]],
            "data_rows": [["B", "11"]],
            "caption_text": "Table 1: Supplementary",
        },
    ]
    (marker_root / "tables_raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 2
    canonical = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "canonical.jsonl"
    payloads = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
    assert len(payloads) == 2


def test_build_structured_exports_merges_adjacent_continued_tables(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "page": 2,
            "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
            "header_rows": [["Polymer", "Value"]],
            "data_rows": [["A", "10"]],
            "caption_text": "Table 2: Data",
        },
        {
            "page": 3,
            "polygons": [[[12, 12], [102, 12], [102, 202], [12, 202]]],
            "header_rows": [["Polymer", "Value"]],
            "data_rows": [["B", "12"]],
            "caption_text": "Table 2 (continued)",
        },
    ]
    (marker_root / "tables_raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 1
    canonical = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "canonical.jsonl"
    payloads = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
    assert len(payloads) == 1
    assert payloads[0]["pages"] == [2, 3]


def test_build_structured_exports_does_not_merge_adjacent_same_number_without_header_similarity(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "page": 2,
            "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
            "header_rows": [["Polymer", "Value"]],
            "data_rows": [["A", "10"]],
            "caption_text": "Table 5: Main run",
        },
        {
            "page": 3,
            "polygons": [[[12, 12], [102, 12], [102, 202], [12, 202]]],
            "header_rows": [["Solvent", "Viscosity"]],
            "data_rows": [["Water", "12"]],
            "caption_text": "Table 5: Alternate set",
        },
    ]
    (marker_root / "tables_raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 2
    canonical = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "canonical.jsonl"
    payloads = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
    assert len(payloads) == 2


def test_build_structured_exports_flattens_multilevel_headers(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_table = {
        "table_group_id": "tblgrp-1",
        "table_block_ids": ["b1"],
        "caption_block_id": "c1",
        "page": 1,
        "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
        "header_rows": [
            ["Polymer", "Solvent", "η = aγ^(n-1)", "η = aγ^(n-1)"],
            ["", "", "a", "n"],
        ],
        "data_rows": [["HPMC", "Water", "1.4E-2", "0.93"]],
        "caption_text": "Table 3",
    }
    (marker_root / "tables_raw.jsonl").write_text(json.dumps(marker_table) + "\n")

    summary = build_structured_exports(doc_dir=doc_dir, table_source="marker-first")
    assert summary.table_count == 1
    table_manifest = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    first_row = json.loads(table_manifest.read_text().splitlines()[0])
    csv_path = doc_dir / first_row["csv_path"]
    header = csv_path.read_text().splitlines()[0]
    assert "η = aγ^(n-1) (a)" in header
    assert "η = aγ^(n-1) (n)" in header


def test_compare_marker_tables_with_ocr_html_reports_symbol_differences(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_row = {
        "table_id": "p0001_t01",
        "page": 1,
        "headers": ["Metric", "Value"],
        "rows": [["surface tension", "25.59 0.14"]],
        "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
    }
    (tables_dir / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")

    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "table_01_page_0001.md").write_text(
        (
            "<table><tr><th>Metric</th><th>Value</th></tr>"
            "<tr><td>surface tension</td><td>25.59 \u00b1 0.14</td></tr></table>"
        )
    )

    report = compare_marker_tables_with_ocr_html(doc_dir=doc_dir)
    assert report["tables_compared"] == 1
    assert report["avg_similarity"] > 0.9
    result = report["results"][0]
    assert "\u00b1" in result["ocr_symbols"]
    assert "\u00b1" in result["missing_in_marker_vs_ocr"]
    assert result["ocr_html_path"].endswith("table_01_page_0001.md")

    report_path = (
        doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_ocr_html_compare.json"
    )
    assert report_path.exists()


def test_compare_marker_tables_parses_html_with_colspan_and_rowspan(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_row = {
        "table_id": "p0001_t01",
        "page": 1,
        "headers": ["Polymer", "Solvent", "eta = a g^(n-1) (a)", "eta = a g^(n-1) (n)"],
        "rows": [["HPMC", "Water", "1.46E-02", "0.93"]],
        "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
    }
    (tables_dir / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")

    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "table_01_page_0001.md").write_text(
        (
            "<table>"
            "<tr><th rowspan='2'>Polymer</th><th rowspan='2'>Solvent</th><th colspan='2'>\\( \\eta = a\\gamma^{n-1} \\)</th></tr>"
            "<tr><th>a</th><th>n</th></tr>"
            "<tr><td>HPMC</td><td>Water</td><td>1.46E-02</td><td>0.93</td></tr>"
            "</table>"
        )
    )

    report = compare_marker_tables_with_ocr_html(doc_dir=doc_dir)
    assert report["tables_compared"] == 1
    result = report["results"][0]
    assert result["ocr_cols"] == 4
    assert "η" in result["ocr_symbols"]
    assert "γ" in result["ocr_symbols"]


def test_compare_marker_tables_matches_by_content_not_ordinal(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "table_id": "p0001_t01",
            "page": 1,
            "headers": ["Material", "δ_D (MPa)1/2"],
            "rows": [["Acetone", "15.5"]],
            "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
        },
        {
            "table_id": "p0001_t02",
            "page": 1,
            "headers": ["Solution", "Measured surface tension"],
            "rows": [["A", "25.59 ± 0.14"]],
            "csv_path": "metadata/assets/structured/extracted/tables/p0001_t02.csv",
        },
    ]
    (tables_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally reversed ordinal content.
    (ocr_dir / "table_01_page_0001.md").write_text(
        "<table><tr><th>Solution</th><th>Measured surface tension</th></tr><tr><td>A</td><td>25.59 ± 0.14</td></tr></table>"
    )
    (ocr_dir / "table_02_page_0001.md").write_text(
        "<table><tr><th>Material</th><th>δ_D (MPa)1/2</th></tr><tr><td>Acetone</td><td>15.5</td></tr></table>"
    )

    report = compare_marker_tables_with_ocr_html(doc_dir=doc_dir)
    assert report["tables_compared"] == 2
    assert report["tables_unmatched_marker"] == 0
    assert report["tables_unmatched_ocr"] == 0
    by_id = {r["table_id"]: r for r in report["results"] if r.get("status") == "ok"}
    assert by_id["p0001_t01"]["similarity"] > 0.9
    assert by_id["p0001_t02"]["similarity"] > 0.9


def test_build_structured_exports_merges_ocr_symbols_into_marker_tables(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    marker_table = {
        "table_group_id": "tblgrp-1",
        "table_block_ids": ["b1"],
        "caption_block_id": "c1",
        "page": 1,
        "polygons": [[[10, 10], [100, 10], [100, 200], [10, 200]]],
        "header_rows": [["Model", "delta_D (MPa)1/2"]],
        "data_rows": [["Power law", "eta = a g^(n-1)"]],
        "caption_text": "Table 1",
    }
    (marker_root / "tables_raw.jsonl").write_text(json.dumps(marker_table) + "\n")

    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "table_01_page_0001.md").write_text(
        (
            "<table><tr><th>Model</th><th>\\( \\delta_D \\) (MPa)<sup>1/2</sup></th></tr>"
            "<tr><td>Power law</td><td>\u03b7 = a\u03b3^(n-1)</td></tr></table>"
        )
    )

    summary = build_structured_exports(
        doc_dir=doc_dir,
        table_source="marker-first",
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
    )
    assert summary.table_count == 1
    assert summary.ocr_merge.get("tables_patched", 0) == 1

    table_manifest = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    first_row = json.loads(table_manifest.read_text().splitlines()[0])
    csv_path = doc_dir / first_row["csv_path"]
    csv_text = csv_path.read_text()
    assert "δ_D (MPa)1/2" in csv_text
    assert "eta = a g^(n-1)" in csv_text

    merge_report = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_ocr_merge.json"
    assert merge_report.exists()


def test_build_structured_exports_merges_ocr_symbols_into_markdown_tables(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "Table 1: Model",
                "| Model | delta_D (MPa)1/2 |",
                "| --- | --- |",
                "| Power law | eta = a g^(n-1) |",
            ]
        )
    )

    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "table_01_page_0001.md").write_text(
        (
            "<table><tr><th>Model</th><th>\\( \\delta_D \\) (MPa)<sup>1/2</sup></th></tr>"
            "<tr><td>Power law</td><td>\u03b7 = a\u03b3^(n-1)</td></tr></table>"
        )
    )

    summary = build_structured_exports(
        doc_dir=doc_dir,
        table_source="marker-first",
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
    )
    assert summary.table_count == 1
    assert summary.ocr_merge.get("mode") == "markdown_tables"
    assert summary.ocr_merge.get("tables_patched", 0) == 1

    csv_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.csv"
    csv_text = csv_path.read_text()
    assert "δ_D (MPa)1/2" in csv_text
    assert "eta = a g^(n-1)" in csv_text


def test_build_structured_exports_strict_mode_requires_marker_artifacts(tmp_path: Path):
    doc_dir = tmp_path / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / "pages" / "0001.md").write_text("No tables\n")
    with pytest.raises(RuntimeError, match="Strict table artifact mode failed"):
        build_structured_exports(
            doc_dir=doc_dir,
            table_source="marker-first",
            table_artifact_mode="strict",
        )
    status_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "pipeline_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text())
    assert payload["status"] == "error"
