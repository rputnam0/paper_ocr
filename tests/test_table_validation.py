from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fitz

from paper_ocr.table_validation import GeminiValidationConfig, run_gemini_table_validation, summarize_gemini_failures


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    async def generate_table_review(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return self._content


class _ErrorClient:
    async def generate_table_review(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("upstream unavailable")


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
                "llm_extraction_instructions": [
                    "Preserve multi-level headers as separate header rows before flattening.",
                    "Resolve alias codes by linking table rows to legend paragraphs in nearby text.",
                ],
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
    assert summary["tables_with_llm_instructions"] == 1
    assert summary["llm_instruction_count"] == 2

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
    assert review["llm_extraction_instructions"] == [
        "Preserve multi-level headers as separate header rows before flattening.",
        "Resolve alias codes by linking table rows to legend paragraphs in nearby text.",
    ]


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


def test_run_gemini_table_validation_continues_on_api_error(tmp_path: Path):
    doc_dir = _make_doc(tmp_path, status="warnings")
    cfg = GeminiValidationConfig(model="gemini-3-flash", api_key="x")

    summary = asyncio.run(
        run_gemini_table_validation(
            ocr_out_dir=tmp_path / "out",
            config=cfg,
            client=_ErrorClient(),
            only_problem_docs=True,
            max_docs=0,
            max_tables_per_doc=0,
        )
    )
    assert summary["tables_reviewed"] == 1
    assert summary["api_error_count"] == 1

    report_path = doc_dir / "metadata" / "assets" / "structured" / "validation" / "gemini_table_review.jsonl"
    rows = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
    assert rows[0]["model_review"]["recommended_action"] == "review"
    assert rows[0]["model_review"]["failure_modes"] == ["other"]


def test_summarize_gemini_failures_aggregates_reports(tmp_path: Path):
    doc_dir = _make_doc(tmp_path, status="warnings")
    validation_dir = doc_dir / "metadata" / "assets" / "structured" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    review_rows = [
        {
            "table_id": "t1",
            "model_review": {
                "recommended_action": "review",
                "failure_modes": ["multi_level_header_loss", "code_legend_unresolved"],
                "rubric": {
                    "formatting_fidelity": 0.8,
                    "structural_fidelity": 0.7,
                    "data_completeness": 0.6,
                    "unit_symbol_fidelity": 0.9,
                    "context_resolution": 0.4,
                    "overall_robustness": "fragile",
                },
            },
        },
        {
            "table_id": "t2",
            "model_review": {
                "recommended_action": "accept",
                "failure_modes": [],
                "rubric": {
                    "formatting_fidelity": 1.0,
                    "structural_fidelity": 1.0,
                    "data_completeness": 1.0,
                    "unit_symbol_fidelity": 1.0,
                    "context_resolution": 1.0,
                    "overall_robustness": "robust",
                },
            },
        },
    ]
    (validation_dir / "gemini_table_review.jsonl").write_text(
        "\n".join(json.dumps(row) for row in review_rows) + "\n"
    )

    out_path = tmp_path / "metrics.json"
    payload = summarize_gemini_failures(ocr_out_dir=tmp_path / "out", report_out=out_path)
    assert payload["tables_reviewed"] == 2
    assert payload["action_counts"]["accept"] == 1
    assert payload["action_counts"]["review"] == 1
    assert payload["failure_mode_counts"]["multi_level_header_loss"] == 1
    assert payload["robustness_counts"]["fragile"] == 1
    assert payload["rubric_averages"]["context_resolution"] == 0.7
    assert out_path.exists()


def test_summarize_gemini_failures_evaluates_gates_against_baseline(tmp_path: Path):
    doc_dir = _make_doc(tmp_path, status="warnings")
    validation_dir = doc_dir / "metadata" / "assets" / "structured" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "gemini_table_review.jsonl").write_text(
        json.dumps(
            {
                "table_id": "t1",
                "model_review": {
                    "recommended_action": "accept",
                    "failure_modes": [],
                    "table_present_on_page": "yes",
                    "rubric": {
                        "formatting_fidelity": 1.0,
                        "structural_fidelity": 1.0,
                        "data_completeness": 1.0,
                        "unit_symbol_fidelity": 1.0,
                        "context_resolution": 1.0,
                        "overall_robustness": "robust",
                    },
                },
            }
        )
        + "\n"
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "action_counts": {"accept": 0, "review": 1, "reject": 1},
                "failure_mode_counts": {"multi_level_header_loss": 2},
                "rubric_averages": {
                    "data_completeness": 0.8,
                    "unit_symbol_fidelity": 0.85,
                    "context_resolution": 0.8,
                },
                "table_present_no": 0,
            }
        )
    )
    gates_path = tmp_path / "gates.json"
    gates_path.write_text(
        json.dumps(
            {
                "global_targets": {
                    "recommended_reject_max": 1,
                    "recommended_accept_delta_min": 1,
                    "table_present_no_max": 0,
                },
                "failure_mode_reduction_targets": {"multi_level_header_loss": 0.5},
                "rubric_targets": {
                    "data_completeness_delta_min": 0.05,
                    "unit_symbol_fidelity_min": 0.9,
                    "context_resolution_delta_min": 0.05,
                },
            }
        )
    )
    payload = summarize_gemini_failures(
        ocr_out_dir=tmp_path / "out",
        report_out=tmp_path / "metrics.json",
        baseline_path=baseline_path,
        gates_path=gates_path,
    )
    assert payload["gate_evaluation"]["available"] is True
    assert payload["gate_evaluation"]["passed"] is True


def test_summarize_gemini_failures_uses_gates_baseline_when_baseline_not_passed(tmp_path: Path):
    _make_doc(tmp_path, status="warnings")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"action_counts": {"accept": 0, "review": 0, "reject": 0}}))
    gates_path = tmp_path / "gates.json"
    gates_path.write_text(json.dumps({"baseline_report": str(baseline_path), "global_targets": {}}))
    payload = summarize_gemini_failures(
        ocr_out_dir=tmp_path / "out",
        report_out=tmp_path / "metrics.json",
        gates_path=gates_path,
    )
    assert payload["baseline_path"] == str(baseline_path)


def test_summarize_gemini_failures_supports_legacy_baseline_shapes(tmp_path: Path):
    doc_dir = _make_doc(tmp_path, status="warnings")
    validation_dir = doc_dir / "metadata" / "assets" / "structured" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "gemini_table_review.jsonl").write_text(
        json.dumps(
            {
                "table_id": "t1",
                "model_review": {
                    "recommended_action": "accept",
                    "failure_modes": ["multi_level_header_loss"],
                    "rubric": {
                        "formatting_fidelity": 1.0,
                        "structural_fidelity": 1.0,
                        "data_completeness": 1.0,
                        "unit_symbol_fidelity": 1.0,
                        "context_resolution": 1.0,
                        "overall_robustness": "robust",
                    },
                },
            }
        )
        + "\n"
    )
    baseline_path = tmp_path / "baseline_legacy.json"
    baseline_path.write_text(
        json.dumps(
            {
                "actions": {"accept": 0, "review": 1, "reject": 1},
                "top_failure_modes": [["multi_level_header_loss", 4]],
            }
        )
    )
    gates_path = tmp_path / "gates.json"
    gates_path.write_text(
        json.dumps(
            {
                "global_targets": {"recommended_accept_delta_min": 1},
                "failure_mode_reduction_targets": {"multi_level_header_loss": 0.5},
            }
        )
    )
    payload = summarize_gemini_failures(
        ocr_out_dir=tmp_path / "out",
        report_out=tmp_path / "metrics.json",
        baseline_path=baseline_path,
        gates_path=gates_path,
    )
    checks = {row["name"]: row for row in payload["gate_evaluation"]["checks"]}
    assert checks["recommended_accept_delta_min"]["actual"] == 1
    assert checks["failure_mode_reduction:multi_level_header_loss"]["baseline_count"] == 4
