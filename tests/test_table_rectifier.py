from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from paper_ocr.table_rectifier import RectifierConfig, run_table_rectification_for_doc


class _FakeTextResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.reasoning_content = None
        self.usage = None
        self.raw = {}


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _make_doc_with_one_table(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    doc_dir = tmp_path / "out" / "group" / "doc_abc"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "0001.md").write_text("Table 1: Polymer data\nLegend A = sample A\n")

    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    qa_dir = doc_dir / "metadata" / "assets" / "structured" / "qa"
    tables_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "table_flags.jsonl").write_text("")

    table_id = "p0001_t01"
    headers = ["Polymer", "Value"]
    rows = [["A", "1.2 ± 0.1"]]
    csv_rel = "metadata/assets/structured/extracted/tables/p0001_t01.csv"
    _write_csv(doc_dir / csv_rel, headers, rows)

    manifest_row = {
        "table_id": table_id,
        "page": 1,
        "caption": "Table 1: Polymer data",
        "headers": headers,
        "rows": rows,
        "csv_path": csv_rel,
    }
    (tables_dir / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")
    table_json = {
        "table_id": table_id,
        "pages": [1],
        "caption_text": "Table 1: Polymer data",
        "header_rows": [headers],
        "header_rows_full": [headers],
        "data_rows": rows,
        "row_lineage": [{"row_index": 0, "fragment_id": "frag-1", "page": 1, "source_row_index": 0, "source_kind": "derived"}],
        "context_mappings": [{"code": "A", "resolved_text": "sample A"}],
        "required_fields_missing": [],
        "quality_metrics": {"empty_cell_ratio": 0.0, "repeated_text_ratio": 0.0, "column_instability_ratio": 0.0},
    }
    (tables_dir / f"{table_id}.json").write_text(json.dumps(table_json, indent=2, ensure_ascii=True))
    return doc_dir, table_json


def _valid_payload() -> str:
    payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "1.2 ± 0.1"]],
        "edits": [{"type": "normalize_header", "description": "No-op normalize"}],
        "cell_provenance": [
            {"row": 0, "col": 0, "source": "context", "evidence_text": "sample A", "confidence": 0.9},
            {"row": 0, "col": 1, "source": "marker", "evidence_text": "1.2 ± 0.1", "confidence": 0.96},
        ],
        "rectifier_confidence": 0.91,
        "needs_review": False,
    }
    return json.dumps(payload)


def test_rectifier_accepts_valid_payload_and_writes_outputs(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )

    assert result.applied == 1
    assert result.fallbacked == 0
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    payload = json.loads(table_json_path.read_text())
    assert payload["rectifier_confidence"] == 0.91
    assert payload["rectifier_needs_review"] is False
    assert payload["llm_rectification"]["applied"] is True
    flags_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_flags.jsonl"
    flags = [json.loads(line) for line in flags_path.read_text().splitlines() if line.strip()]
    assert any(flag["type"] == "llm_rectification_applied" for flag in flags)


def test_rectifier_recovers_from_malformed_json_with_repair_call(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    calls = {"count": 0}

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeTextResponse("```json\n{\"rectified_header_rows_full\":")
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert calls["count"] == 2
    assert result.applied == 1
    assert result.invalid_schema == 0


def test_rectifier_rejects_evidence_violation_and_falls_back(tmp_path: Path):
    doc_dir, before = _make_doc_with_one_table(tmp_path)
    payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "FABRICATED 99"]],
        "edits": [{"type": "replace_cell", "description": "invented value"}],
        "cell_provenance": [
            {"row": 0, "col": 0, "source": "context", "evidence_text": "sample A", "confidence": 0.9},
            {"row": 0, "col": 1, "source": "context", "evidence_text": "fabricated", "confidence": 0.95},
        ],
        "rectifier_confidence": 0.8,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )

    assert result.applied == 0
    assert result.fallbacked == 1
    assert result.evidence_violations == 1
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    after = json.loads(table_json_path.read_text())
    assert after["data_rows"] == before["data_rows"]
    flags_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_flags.jsonl"
    flags = [json.loads(line) for line in flags_path.read_text().splitlines() if line.strip()]
    assert any(flag["type"] == "llm_rectification_evidence_violation" for flag in flags)


def test_rectifier_requires_provenance_for_new_nonempty_cells(tmp_path: Path):
    doc_dir, before = _make_doc_with_one_table(tmp_path)
    payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "1.2 ± 0.1"], ["B", "2.0"]],
        "edits": [{"type": "add_row", "description": "new row from context"}],
        "cell_provenance": [
            {"row": 0, "col": 0, "source": "context", "evidence_text": "sample A", "confidence": 0.9},
            {"row": 0, "col": 1, "source": "marker", "evidence_text": "1.2 ± 0.1", "confidence": 0.95},
        ],
        "rectifier_confidence": 0.75,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )

    assert result.applied == 0
    assert result.evidence_violations == 1
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    after = json.loads(table_json_path.read_text())
    assert after["data_rows"] == before["data_rows"]


def test_rectifier_infers_missing_provenance_from_ocr_evidence(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    ocr_dir = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "table_01_page_0001.md").write_text(
        "<table><tr><th>Polymer</th><th>Value</th></tr><tr><td>A</td><td>1.2 ± 0.2</td></tr></table>"
    )
    payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "1.2 ± 0.2"]],
        "edits": [{"type": "replace_cell", "description": "fix OCR symbol"}],
        "cell_provenance": [],
        "rectifier_confidence": 0.82,
        "needs_review": False,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert result.applied == 1
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    after = json.loads(table_json_path.read_text())
    assert after["data_rows"] == [["A", "1.2 ± 0.2"]]
    assert any(int(item.get("row", -1)) == 0 and int(item.get("col", -1)) == 1 for item in after["cell_provenance"])


def test_rectifier_retries_on_invalid_schema_before_fallback(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    calls = {"count": 0}
    invalid_payload = {
        "rectified_header_rows_full": [],
        "rectified_rows": [],
        "edits": [],
        "cell_provenance": [],
        "rectifier_confidence": 0.2,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeTextResponse(json.dumps(invalid_payload))
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert calls["count"] == 2
    assert result.applied == 1


def test_rectifier_repairs_symbol_loss_by_restoring_original_header_symbol(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    table_payload = json.loads(table_json_path.read_text())
    table_payload["header_rows_full"] = [["Polymer", "η (mPa·s)"]]
    table_payload["data_rows"] = [["A", "1.2 ± 0.1"]]
    table_json_path.write_text(json.dumps(table_payload, indent=2, ensure_ascii=True))
    manifest_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().strip())
    manifest["headers"] = ["Polymer", "η (mPa·s)"]
    manifest["rows"] = [["A", "1.2 ± 0.1"]]
    manifest_path.write_text(json.dumps(manifest) + "\n")

    payload = {
        "rectified_header_rows_full": [["Polymer", "(mPa·s)"]],
        "rectified_rows": [["A", "1.2 ± 0.1"]],
        "edits": [{"type": "header_cleanup", "description": "drop greek symbol"}],
        "cell_provenance": [],
        "rectifier_confidence": 0.7,
        "needs_review": False,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert result.applied == 1
    after = json.loads(table_json_path.read_text())
    assert "η" in after["header_rows_full"][0][1]


def test_rectifier_repairs_row_loss_by_restoring_missing_rows(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    table_payload = json.loads(table_json_path.read_text())
    table_payload["data_rows"] = [["A", "1.2 ± 0.1"], ["B", "2.4 ± 0.2"]]
    table_json_path.write_text(json.dumps(table_payload, indent=2, ensure_ascii=True))

    manifest_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().strip())
    manifest["rows"] = [["A", "1.2 ± 0.1"], ["B", "2.4 ± 0.2"]]
    manifest_path.write_text(json.dumps(manifest) + "\n")

    payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "1.2 ± 0.1"]],
        "edits": [{"type": "dedupe_rows", "description": "incorrectly dropped row"}],
        "cell_provenance": [],
        "rectifier_confidence": 0.6,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert result.applied == 1
    after = json.loads(table_json_path.read_text())
    assert len(after["data_rows"]) == 2
