from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fitz

from paper_ocr.table_validation import GeminiValidationConfig, run_gemini_table_validation


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def generate_table_review(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        return self._content


def _make_doc(tmp_path: Path, *, status: str = "warnings") -> Path:
    doc_dir = tmp_path / "out" / "group" / "doc_abc"
    (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables").mkdir(parents=True, exist_ok=True)
    (doc_dir / "metadata" / "assets" / "structured" / "qa").mkdir(parents=True, exist_ok=True)
    source_pdf = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=600)
        page.insert_text((72, 72), "Table 1: Polymer data")
        pdf.save(source_pdf)
    (doc_dir / "metadata" / "manifest.json").write_text(json.dumps({"source_path": str(source_pdf)}))
    (doc_dir / "metadata" / "assets" / "structured" / "qa" / "pipeline_status.json").write_text(
        json.dumps({"status": status, "errors": ["marker_tables_raw_missing"] if status != "ok" else []})
    )
    table_row = {
        "table_id": "p0001_t01",
        "page": 1,
        "caption": "Table 1: Polymer data",
        "headers": ["Polymer", "Mw"],
        "rows": [["A", "12000"], ["B", "32000"]],
        "csv_path": "metadata/assets/structured/extracted/tables/p0001_t01.csv",
    }
    (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl").write_text(
        json.dumps(table_row) + "\n"
    )
    return doc_dir


def test_run_gemini_table_validation_writes_doc_report(tmp_path: Path):
    doc_dir = _make_doc(tmp_path, status="warnings")
    fake_client = _FakeClient(
        json.dumps(
            {
                "table_present_on_page": "yes",
                "extraction_quality": 0.93,
                "false_positive_risk": 0.04,
                "recommended_action": "accept",
                "issues": [],
            }
        )
    )
    cfg = GeminiValidationConfig(model="gemini-3-flash", api_key="x")

    summary = asyncio.run(
        run_gemini_table_validation(
            ocr_out_dir=tmp_path / "out",
            config=cfg,
            client=fake_client,
            only_problem_docs=True,
            max_docs=0,
            max_tables_per_doc=0,
        )
    )

    assert summary["docs_reviewed"] == 1
    assert summary["tables_reviewed"] == 1
    assert summary["table_present_yes"] == 1
    report_path = doc_dir / "metadata" / "assets" / "structured" / "validation" / "gemini_table_review.jsonl"
    assert report_path.exists()
    rows = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
    assert rows[0]["model_review"]["recommended_action"] == "accept"
    assert fake_client.calls == 1


def test_run_gemini_table_validation_skips_ok_docs_when_problem_filter_enabled(tmp_path: Path):
    _make_doc(tmp_path, status="ok")
    fake_client = _FakeClient("{}")
    cfg = GeminiValidationConfig(model="gemini-3-flash", api_key="x")

    summary = asyncio.run(
        run_gemini_table_validation(
            ocr_out_dir=tmp_path / "out",
            config=cfg,
            client=fake_client,
            only_problem_docs=True,
            max_docs=0,
            max_tables_per_doc=0,
        )
    )
    assert summary["docs_reviewed"] == 0
    assert summary["tables_reviewed"] == 0
    assert fake_client.calls == 0


def test_run_gemini_table_validation_parses_partial_json(tmp_path: Path):
    _make_doc(tmp_path, status="warnings")
    fake_client = _FakeClient('```json\n{"table_present_on_page":"yes","extraction_quality":0.81,"false_')
    cfg = GeminiValidationConfig(model="gemini-3-flash", api_key="x")

    summary = asyncio.run(
        run_gemini_table_validation(
            ocr_out_dir=tmp_path / "out",
            config=cfg,
            client=fake_client,
            only_problem_docs=True,
            max_docs=0,
            max_tables_per_doc=1,
        )
    )
    assert summary["tables_reviewed"] == 1
    assert summary["table_present_yes"] == 1
