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


def _valid_payload_with_context_request(*, window: str, reason: str) -> str:
    payload = json.loads(_valid_payload())
    payload["context_request"] = {
        "needed": True,
        "window": window,
        "reason": reason,
    }
    return json.dumps(payload)


def _extract_prompt_input(prompt: str) -> dict[str, object]:
    marker = "Input:\n"
    if marker not in prompt:
        return {}
    raw = prompt.split(marker, 1)[1]
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def test_rectifier_skips_already_rectified_when_enabled(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    payload = json.loads(table_json_path.read_text())
    payload["llm_rectification"] = {"applied": True, "model": "openai/gpt-oss-120b"}
    table_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    calls = {"count": 0}

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all", skip_already_rectified=True),
            call_model=_fake_call_model,
        )
    )
    assert calls["count"] == 0
    assert result.selected == 0
    assert result.applied == 0
    assert any(row.get("status") == "skipped_already_rectified" for row in result.table_results)


def test_rectifier_rerectifies_already_rectified_by_default(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    payload = json.loads(table_json_path.read_text())
    payload["llm_rectification"] = {"applied": True, "model": "openai/gpt-oss-120b"}
    table_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    calls = {"count": 0}

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all"),
            call_model=_fake_call_model,
        )
    )
    assert calls["count"] == 1
    assert result.applied == 1


def test_rectifier_uses_original_snapshot_as_fresh_input(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    payload = json.loads(table_json_path.read_text())
    payload["data_rows"] = [["A", "9.9"]]
    payload["llm_rectification"] = {
        "applied": True,
        "model": "openai/gpt-oss-120b",
        "original_snapshot": {
            "header_rows_full": [["Polymer", "Value"]],
            "data_rows": [["A", "1.2 \u00b1 0.1"]],
            "header_hierarchy": [],
            "row_lineage": payload.get("row_lineage", []),
            "context_mappings": [],
            "required_fields_missing": [],
            "quality_metrics": payload.get("quality_metrics", {}),
        },
    }
    table_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    prompts: list[str] = []

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        prompts.append(str(kwargs.get("prompt", "")))
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
    assert prompts
    payload_in = _extract_prompt_input(prompts[0])
    canonical = payload_in.get("canonical_table", {}) if isinstance(payload_in.get("canonical_table"), dict) else {}
    rows = canonical.get("data_rows", [])
    assert rows == [["A", "1.2 \u00b1 0.1"]]


def test_rectifier_writes_run_scoped_canonical_snapshots(tmp_path: Path):
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
    assert result.run_id
    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    input_snapshot = tables_dir / f"canonical.input.{result.run_id}.jsonl"
    output_snapshot = tables_dir / f"canonical.rectified.{result.run_id}.jsonl"
    assert input_snapshot.exists()
    assert output_snapshot.exists()
    in_rows = [json.loads(line) for line in input_snapshot.read_text().splitlines() if line.strip()]
    out_rows = [json.loads(line) for line in output_snapshot.read_text().splitlines() if line.strip()]
    assert in_rows and out_rows
    assert in_rows[0]["table_id"] == "p0001_t01"
    assert out_rows[0]["table_id"] == "p0001_t01"


def test_rectifier_requests_nearby_context_only_on_demand(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    prompts: list[str] = []
    calls = {"count": 0}

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        prompts.append(str(kwargs.get("prompt", "")))
        if calls["count"] == 1:
            return _FakeTextResponse(
                _valid_payload_with_context_request(
                    window="table_page",
                    reason="aliases_or_abbreviations_without_definition",
                )
            )
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all", context_mode="on_demand"),
            call_model=_fake_call_model,
        )
    )

    assert result.applied == 1
    assert calls["count"] == 2
    first = _extract_prompt_input(prompts[0])
    second = _extract_prompt_input(prompts[1])
    first_evidence = first.get("evidence", {}) if isinstance(first.get("evidence"), dict) else {}
    second_evidence = second.get("evidence", {}) if isinstance(second.get("evidence"), dict) else {}
    assert first_evidence.get("nearby_pages_text", "") == ""
    assert "Legend A = sample A" in str(second_evidence.get("nearby_pages_text", ""))
    assert result.table_results and result.table_results[0].get("context_requested") is True
    assert result.table_results[0].get("context_window") == "table_page"


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
        "rectified_rows": [["A", "FABRICATED ± 99"]],
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


def test_rectifier_retries_after_evidence_violation_with_validation_feedback(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    validation_dir = doc_dir / "metadata" / "assets" / "structured" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_row = {
        "table_id": "p0001_t01",
        "model_review": {
            "recommended_action": "review",
            "failure_modes": ["row_shift_or_merge", "symbol_or_unit_loss"],
            "issues": ["Mean and std-dev were split across rows."],
            "llm_extraction_instructions": ["Merge vertically stacked continuation values into a single logical row."],
            "final_action": "reject",
        },
    }
    (validation_dir / "gemini_table_review.jsonl").write_text(json.dumps(validation_row) + "\n")

    calls = {"count": 0}
    prompts: list[str] = []

    bad_payload = {
        "rectified_header_rows_full": [["Polymer", "Value"]],
        "rectified_rows": [["A", "FABRICATED ± 99"]],
        "edits": [{"type": "replace_cell", "description": "hallucinated replacement"}],
        "cell_provenance": [{"row": 0, "col": 1, "source": "context", "evidence_text": "fabricated", "confidence": 0.9}],
        "rectifier_confidence": 0.6,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        calls["count"] += 1
        prompts.append(str(kwargs.get("prompt", "")))
        if calls["count"] == 1:
            return _FakeTextResponse(json.dumps(bad_payload))
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
    assert calls["count"] == 2
    assert any("Merge vertically stacked continuation values" in prompt for prompt in prompts)


def test_rectifier_target_reject_uses_validation_final_action(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    validation_dir = doc_dir / "metadata" / "assets" / "structured" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_row = {
        "table_id": "p0001_t01",
        "model_review": {
            "recommended_action": "review",
            "final_action": "reject",
            "failure_modes": ["multi_level_header_loss"],
        },
    }
    (validation_dir / "gemini_table_review.jsonl").write_text(json.dumps(validation_row) + "\n")

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="reject"),
            call_model=_fake_call_model,
        )
    )
    assert result.selected == 1
    assert result.applied == 1


def test_rectifier_preserves_multilevel_header_rows_full(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    table_payload = json.loads(table_json_path.read_text())
    table_payload["header_rows_full"] = [["Condition", "Condition"], ["Polymer", "Value"]]
    table_payload["data_rows"] = [["A", "1.2 ± 0.1"]]
    table_json_path.write_text(json.dumps(table_payload, indent=2, ensure_ascii=True))

    payload = {
        "rectified_header_rows_full": [["Condition", "Condition"], ["Polymer", "Value"]],
        "rectified_rows": [["A", "1.2 ± 0.1"]],
        "edits": [{"type": "preserve", "description": "keep hierarchical headers"}],
        "cell_provenance": [],
        "rectifier_confidence": 0.9,
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
    assert len(after["header_rows_full"]) == 2


def _write_table_structure_artifact(doc_dir: Path, table_id: str, *, rows: int, cols: int, header_rows: int = 1) -> None:
    structure_root = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_structure"
    structure_root.mkdir(parents=True, exist_ok=True)
    structure_payload = {
        "table_id": table_id,
        "model": "tatr",
        "rows": rows,
        "cols": cols,
        "header_rows": header_rows,
        "cells": [],
        "status": "ok",
    }
    structure_path = structure_root / f"{table_id}.json"
    structure_path.write_text(json.dumps(structure_payload, indent=2, ensure_ascii=True))
    manifest_row = {
        "table_id": table_id,
        "status": "ok",
        "model": "tatr",
        "structure_path": f"metadata/assets/structured/qa/table_structure/{table_id}.json",
    }
    (structure_root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n")


def test_rectifier_includes_table_structure_in_prompt(tmp_path: Path):
    doc_dir, _ = _make_doc_with_one_table(tmp_path)
    _write_table_structure_artifact(doc_dir, "p0001_t01", rows=2, cols=2, header_rows=1)
    prompts: list[str] = []

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        prompts.append(str(kwargs.get("prompt", "")))
        return _FakeTextResponse(_valid_payload())

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all", structure_model="tatr", structure_lock=True),
            call_model=_fake_call_model,
        )
    )
    assert result.applied == 1
    joined = "\n".join(prompts)
    assert '"table_structure"' in joined
    assert '"cols": 2' in joined


def test_rectifier_enforces_structure_lock_when_tatr_structure_present(tmp_path: Path):
    doc_dir, before = _make_doc_with_one_table(tmp_path)
    _write_table_structure_artifact(doc_dir, "p0001_t01", rows=2, cols=2, header_rows=1)
    payload = {
        "rectified_header_rows_full": [["Polymer", "Value", "Extra"]],
        "rectified_rows": [["A", "1.2 ± 0.1", "X"]],
        "edits": [{"type": "add_col", "description": "violates locked structure"}],
        "cell_provenance": [
            {"row": 0, "col": 0, "source": "context", "evidence_text": "sample A", "confidence": 0.9},
            {"row": 0, "col": 1, "source": "marker", "evidence_text": "1.2 ± 0.1", "confidence": 0.9},
            {"row": 0, "col": 2, "source": "context", "evidence_text": "x", "confidence": 0.9},
        ],
        "rectifier_confidence": 0.7,
        "needs_review": True,
    }

    async def _fake_call_model(**kwargs):  # noqa: ANN001
        return _FakeTextResponse(json.dumps(payload))

    result = asyncio.run(
        run_table_rectification_for_doc(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            config=RectifierConfig(target="all", structure_model="tatr", structure_lock=True),
            call_model=_fake_call_model,
        )
    )
    assert result.applied == 0
    assert result.fallbacked == 1
    assert result.evidence_violations == 1
    table_json_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "p0001_t01.json"
    after = json.loads(table_json_path.read_text())
    assert after["data_rows"] == before["data_rows"]
