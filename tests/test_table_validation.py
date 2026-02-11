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
        self.last_kwargs: dict[str, object] = {}

    async def generate_table_review(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return self._content


def _make_doc(tmp_path: Path, *, status: str = "warnings", continuation_pages: list[int] | None = None) -> Path:
    doc_dir = tmp_path / "out" / "group" / "doc_abc"
    (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables").mkdir(parents=True, exist_ok=True)
    (doc_dir / "metadata" / "assets" / "structured" / "qa").mkdir(parents=True, exist_ok=True)
    source_pdf = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=600)
        page.insert_text((72, 72), "Table 1: Polymer data")
        if continuation_pages and 2 in continuation_pages:
            page2 = pdf.new_page(width=400, height=600)
            page2.insert_text((72, 72), "Table 1 continued")
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
    if continuation_pages:
        table_detail = {
            "table_id": table_row["table_id"],
            "pages": continuation_pages,
            "caption_text": table_row["caption"],
            "header_rows": [table_row["headers"]],
            "data_rows": table_row["rows"],
        }
        (doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / f"{table_row['table_id']}.json").write_text(
            json.dumps(table_detail)
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
    assert rows[0]["model_review"]["rubric"]["overall_robustness"] in {"robust", "mostly_robust", "fragile", "failed"}
    assert fake_client.calls == 1


def test_run_gemini_table_validation_tracks_rubric_and_failure_modes(tmp_path: Path):
    _make_doc(tmp_path, status="warnings")
    fake_client = _FakeClient(
        json.dumps(
            {
                "table_present_on_page": "yes",
                "extraction_quality": 0.62,
                "false_positive_risk": 0.18,
                "recommended_action": "review",
                "issues": ["Alias polymer codes are unresolved"],
                "failure_modes": ["code_legend_unresolved", "missing_required_columns"],
                "root_cause_hypothesis": "Legend mapping was not merged into table output.",
                "needs_followup": True,
                "followup_recommendations": ["Resolve polymer alias codes from nearby text before export."],
                "missing_required_information": ["Polymer code legend mapping"],
                "formatting_issues": ["Unit superscript formatting dropped in two headers."],
                "rubric": {
                    "formatting_fidelity": 0.7,
                    "structural_fidelity": 0.65,
                    "data_completeness": 0.6,
                    "unit_symbol_fidelity": 0.5,
                    "context_resolution": 0.2,
                    "overall_robustness": "fragile",
                },
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
    assert summary["tables_reviewed"] == 1
    assert summary["needs_followup_count"] == 1
    assert summary["failure_mode_counts"]["code_legend_unresolved"] == 1
    assert summary["failure_mode_counts"]["missing_required_columns"] == 1
    assert summary["robustness_counts"]["fragile"] == 1
    assert summary["rubric_averages"]["context_resolution"] == 0.2

    report_path = (
        tmp_path
        / "out"
        / "group"
        / "doc_abc"
        / "metadata"
        / "assets"
        / "structured"
        / "validation"
        / "gemini_table_review.jsonl"
    )
    rows = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
    review = rows[0]["model_review"]
    assert review["missing_required_information"] == ["Polymer code legend mapping"]
    assert review["formatting_issues"] == ["Unit superscript formatting dropped in two headers."]


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


def test_run_gemini_table_validation_uses_continuation_pages_when_available(tmp_path: Path):
    _make_doc(tmp_path, status="warnings", continuation_pages=[1, 2])
    fake_client = _FakeClient(
        json.dumps(
            {
                "table_present_on_page": "yes",
                "extraction_quality": 0.9,
                "false_positive_risk": 0.05,
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
    assert summary["tables_reviewed"] == 1
    image_list = fake_client.last_kwargs.get("image_bytes_list")
    assert isinstance(image_list, list)
    assert len(image_list) == 2
    page_numbers = fake_client.last_kwargs.get("page_numbers")
    assert page_numbers == [1, 2]
