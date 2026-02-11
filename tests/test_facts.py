from __future__ import annotations

import json
from pathlib import Path

from paper_ocr.facts import export_facts_for_doc, validate_property_record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")


def test_export_facts_for_doc_writes_records_and_manifest(tmp_path: Path):
    doc_dir = tmp_path / "group" / "doc_abc"
    metadata = doc_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "manifest.json").write_text(json.dumps({"doc_id": "abc"}))
    table_rows = [
        {
            "table_id": "p0001_t01",
            "page": 1,
            "caption": "Table 1",
            "headers": ["Material", "Viscosity (Pa s)"],
            "rows": [["PVP", "1.20"]],
            "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
        }
    ]
    _write_jsonl(metadata / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl", table_rows)
    _write_jsonl(
        metadata / "assets" / "structured" / "extracted" / "figures" / "manifest.jsonl",
        [{"figure_id": "p0001_f01", "page": 1, "alt_text": "rheology curve", "image_ref": "_fig1.png"}],
    )

    summary = export_facts_for_doc(doc_dir)
    assert summary["record_count"] >= 2
    records_path = doc_dir / "metadata" / "assets" / "structured" / "facts" / "property_records.jsonl"
    assert records_path.exists()
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    assert records
    assert all(validate_property_record(r) == [] for r in records)
    assert all("source" in r for r in records)
    assert all(0.0 <= float(r["confidence"]) <= 1.0 for r in records)

    manifest_path = doc_dir / "metadata" / "assets" / "structured" / "facts" / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["record_count"] == summary["record_count"]


def test_export_facts_updates_document_manifest(tmp_path: Path):
    doc_dir = tmp_path / "group" / "doc_abc"
    metadata = doc_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata / "manifest.json"
    manifest_path.write_text(json.dumps({"doc_id": "abc"}))
    _write_jsonl(
        metadata / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl",
        [
            {
                "table_id": "p0001_t01",
                "page": 1,
                "caption": "Table 1",
                "headers": ["Material", "Value"],
                "rows": [["A", "10"]],
                "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
            }
        ],
    )

    export_facts_for_doc(doc_dir)
    manifest = json.loads(manifest_path.read_text())
    assert "facts_extraction" in manifest
    facts = manifest["facts_extraction"]
    assert facts["enabled"] is True
    assert facts["record_count"] >= 1
    assert facts["records_path"].endswith("property_records.jsonl")
