from __future__ import annotations

from pathlib import Path

from paper_ocr.table_context import resolve_table_context_mappings


def test_resolve_table_context_mappings_links_alias_codes_from_nearby_text(tmp_path: Path):
    doc_dir = tmp_path / "doc_abc"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text(
        "\n".join(
            [
                "Materials legend:",
                "P1 = poly(styrene)",
                "P2 = poly(vinyl alcohol)",
            ]
        )
    )

    table = {
        "headers": ["Polymer code", "Viscosity (Pa s)"],
        "rows": [["P1", "12.3"], ["P2", "8.1"]],
    }
    out = resolve_table_context_mappings(doc_dir=doc_dir, page=1, table=table)

    resolved = {row["code"]: row["resolved_text"] for row in out["mappings"]}
    assert resolved["P1"] == "poly(styrene)"
    assert resolved["P2"] == "poly(vinyl alcohol)"
    assert out["unresolved_codes"] == []


def test_resolve_table_context_mappings_tracks_unresolved_codes(tmp_path: Path):
    doc_dir = tmp_path / "doc_abc"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "0001.md").write_text("A1 = reference blend\n")

    table = {
        "headers": ["Sample", "Mw"],
        "rows": [["A1", "100"], ["B2", "120"]],
    }
    out = resolve_table_context_mappings(doc_dir=doc_dir, page=1, table=table)

    resolved = {row["code"] for row in out["mappings"]}
    assert "A1" in resolved
    assert "B2" in out["unresolved_codes"]

