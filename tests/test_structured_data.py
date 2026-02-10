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

    monkeypatch.setattr("subprocess.run", _fake_run)

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
