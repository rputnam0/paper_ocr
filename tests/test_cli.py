import argparse
import asyncio
import json
from pathlib import Path

import fitz
import pytest

from paper_ocr import cli
from paper_ocr.inspect import TextHeuristics
from paper_ocr.structured_data import StructuredExportSummary
from paper_ocr.structured_extract import StructuredPageResult


class _FakeDoc:
    def __init__(self, page_count: int) -> None:
        self._page_count = page_count
        self._closed = False

    @property
    def page_count(self) -> int:
        if self._closed:
            raise ValueError("document closed")
        return self._page_count


class _FakeOpen:
    def __init__(self, page_count: int) -> None:
        self._doc = _FakeDoc(page_count)

    def __enter__(self) -> _FakeDoc:
        return self._doc

    def __exit__(self, exc_type, exc, tb) -> None:
        self._doc._closed = True


async def _fake_extract_discovery(*args, **kwargs):
    return {"paper_summary": "", "key_topics": [], "sections": []}


async def _fake_fetch_from_telegram(config):
    _fake_fetch_from_telegram.last_config = config
    return []


def test_process_pdf_uses_cached_page_count_after_close(monkeypatch, tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    pdf_path = in_dir / "doc.pdf"
    pdf_path.write_bytes(b"pdf")

    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        out_dir=out_dir,
        workers=1,
        model="m",
        base_url="https://example.com/v1/openai",
        max_tokens=32,
        force=False,
        mode="auto",
        debug=False,
        scan_preprocess=False,
        text_only=False,
        metadata_model="meta",
    )

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123")
    monkeypatch.setattr(cli.fitz, "open", lambda p: _FakeOpen(page_count=0))
    monkeypatch.setattr(cli, "_extract_discovery", _fake_extract_discovery)

    result = asyncio.run(cli._process_pdf(args, pdf_path))

    assert result["page_count"] == 0


def test_parse_fetch_telegram_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "fetch-telegram",
            "papers.csv",
            "data/jobs",
            "--doi-column",
            "DOI",
        ],
    )

    args = cli._parse_args()

    assert args.command == "fetch-telegram"
    assert args.doi_column == "DOI"
    assert args.doi_csv == Path("papers.csv")
    assert args.output_root == Path("data/jobs")


def test_parse_fetch_telegram_defaults(monkeypatch):
    monkeypatch.delenv("MIN_DELAY", raising=False)
    monkeypatch.delenv("MAX_DELAY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "fetch-telegram",
            "papers.csv",
        ],
    )

    args = cli._parse_args()

    assert args.output_root == Path("data/jobs")
    assert args.min_delay == 4.0
    assert args.max_delay == 8.0
    assert args.response_timeout == 15
    assert args.search_timeout == 40


def test_parse_export_structured_data_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "export-structured-data",
            "out",
            "--deplot-command",
            "deplot-cli --image {image}",
            "--deplot-timeout",
            "45",
        ],
    )
    args = cli._parse_args()
    assert args.command == "export-structured-data"
    assert args.ocr_out_dir == Path("out")
    assert args.deplot_command == "deplot-cli --image {image}"
    assert args.deplot_timeout == 45


def test_parse_run_table_pipeline_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "run",
            "data/in",
            "out",
        ],
    )
    args = cli._parse_args()
    assert args.marker_localize is True
    assert args.marker_localize_profile == "full_json"
    assert args.layout_fallback == "surya"
    assert args.table_source == "marker-first"
    assert args.table_ocr_merge is True
    assert args.table_ocr_merge_scope == "header"
    assert args.table_header_ocr_auto is True
    assert args.table_artifact_mode == "permissive"
    assert args.table_quality_gate is True
    assert args.table_escalation == "auto"
    assert args.table_escalation_max == 20
    assert args.table_qa_mode == "warn"
    assert args.compare_ocr_html is False
    assert args.ocr_html_dir is None


def test_parse_export_table_pipeline_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "export-structured-data",
            "out",
            "--table-source",
            "markdown-only",
            "--table-qa-mode",
            "strict",
            "--table-escalation",
            "always",
            "--table-escalation-max",
            "3",
            "--no-table-ocr-merge",
            "--table-ocr-merge-scope",
            "full",
            "--no-table-header-ocr-auto",
            "--table-artifact-mode",
            "strict",
        ],
    )
    args = cli._parse_args()
    assert args.table_source == "markdown-only"
    assert args.table_qa_mode == "strict"
    assert args.table_escalation == "always"
    assert args.table_escalation_max == 3
    assert args.table_ocr_merge is False
    assert args.table_ocr_merge_scope == "full"
    assert args.table_header_ocr_auto is False
    assert args.table_artifact_mode == "strict"
    assert args.compare_ocr_html is False
    assert args.ocr_html_dir is None


def test_parse_export_table_comparison_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "export-structured-data",
            "out",
            "--compare-ocr-html",
            "--ocr-html-dir",
            "custom/ocr_html",
        ],
    )
    args = cli._parse_args()
    assert args.table_ocr_merge is True
    assert args.table_ocr_merge_scope == "header"
    assert args.table_header_ocr_auto is True
    assert args.table_artifact_mode == "permissive"
    assert args.compare_ocr_html is True
    assert args.ocr_html_dir == Path("custom/ocr_html")


def test_parse_eval_table_pipeline_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "eval-table-pipeline",
            "gold",
            "pred",
        ],
    )
    args = cli._parse_args()
    assert args.command == "eval-table-pipeline"
    assert args.gold_dir == Path("gold")
    assert args.pred_dir == Path("pred")


def test_parse_data_audit_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "data-audit",
            "data",
            "--strict",
            "--json",
        ],
    )
    args = cli._parse_args()
    assert args.command == "data-audit"
    assert args.data_dir == Path("data")
    assert args.strict is True
    assert args.json is True


def test_run_data_audit_strict_raises(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "unexpected").mkdir(parents=True)
    args = argparse.Namespace(data_dir=data_dir, strict=True, json=False)
    with pytest.raises(SystemExit, match="Data layout contract violations"):
        cli._run_data_audit(args)


def test_run_data_audit_non_strict_returns_report(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    args = argparse.Namespace(data_dir=data_dir, strict=False, json=False)
    report = cli._run_data_audit(args)
    assert report["issue_count"] == 0
    assert report["data_dir"] == str(data_dir.resolve())


def test_run_export_structured_data_updates_manifest(monkeypatch, tmp_path: Path):
    root = tmp_path / "out"
    doc_dir = root / "group" / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / "metadata").mkdir(parents=True)
    manifest_path = doc_dir / "metadata" / "manifest.json"
    manifest_path.write_text(json.dumps({"doc_id": "abc"}))

    seen_kwargs: dict[str, object] = {}

    def _fake_build(**kwargs):  # noqa: ANN001
        seen_kwargs.update(kwargs)
        return StructuredExportSummary(
            table_count=2,
            figure_count=3,
            deplot_count=1,
            unresolved_figure_count=0,
            errors=[],
        )

    monkeypatch.setattr(cli, "build_structured_exports", _fake_build)
    monkeypatch.setattr(
        cli,
        "compare_marker_tables_with_ocr_html",
        lambda **kwargs: {
            "tables_compared": 2,
            "avg_similarity": 0.95,
            "report_path": "metadata/assets/structured/qa/table_ocr_html_compare.json",
        },
    )

    args = argparse.Namespace(
        ocr_out_dir=root,
        deplot_command="deplot-cli --image {image}",
        deplot_timeout=30,
        table_source="marker-first",
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
        table_header_ocr_auto=False,
        table_header_ocr_model="m",
        table_header_ocr_max_tokens=500,
        table_artifact_mode="permissive",
        ocr_html_dir=None,
        table_quality_gate=True,
        table_escalation="auto",
        table_escalation_max=20,
        table_qa_mode="warn",
        compare_ocr_html=True,
    )
    result = cli._run_export_structured_data(args)
    assert result["docs_processed"] == 1
    payload = json.loads(manifest_path.read_text())
    assert payload["structured_data_extraction"]["table_count"] == 2
    assert payload["structured_data_extraction"]["figure_count"] == 3
    assert payload["structured_data_extraction"]["ocr_html_comparison"]["tables_compared"] == 2
    assert seen_kwargs["table_ocr_merge"] is True
    assert seen_kwargs["table_ocr_merge_scope"] == "header"
    assert seen_kwargs["table_artifact_mode"] == "permissive"
    assert seen_kwargs["ocr_html_dir"] is None
    assert seen_kwargs["grobid_status"] == "unknown"


def test_run_export_structured_data_passes_grobid_ok_when_manifest_records_usage(monkeypatch, tmp_path: Path):
    root = tmp_path / "out"
    doc_dir = root / "group" / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / "metadata").mkdir(parents=True)
    manifest_path = doc_dir / "metadata" / "manifest.json"
    manifest_path.write_text(json.dumps({"structured_extraction": {"grobid_used": True}}))
    seen_kwargs: dict[str, object] = {}

    def _fake_build(**kwargs):  # noqa: ANN001
        seen_kwargs.update(kwargs)
        return StructuredExportSummary()

    monkeypatch.setattr(cli, "build_structured_exports", _fake_build)

    args = argparse.Namespace(
        ocr_out_dir=root,
        deplot_command="",
        deplot_timeout=30,
        table_source="marker-first",
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
        table_header_ocr_auto=False,
        table_header_ocr_model="m",
        table_header_ocr_max_tokens=500,
        table_artifact_mode="permissive",
        ocr_html_dir=None,
        table_quality_gate=True,
        table_escalation="auto",
        table_escalation_max=20,
        table_qa_mode="warn",
    )
    cli._run_export_structured_data(args)
    assert seen_kwargs["grobid_status"] == "ok"


def test_run_export_structured_data_requires_docs(tmp_path: Path):
    args = argparse.Namespace(
        ocr_out_dir=tmp_path / "empty_out",
        deplot_command="",
        deplot_timeout=90,
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
        table_header_ocr_auto=False,
        table_header_ocr_model="m",
        table_header_ocr_max_tokens=500,
        table_artifact_mode="permissive",
        ocr_html_dir=None,
    )
    args.ocr_out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="No OCR document folders found"):
        cli._run_export_structured_data(args)


def test_run_text_only_enabled_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "run",
            "data/in",
            "out",
        ],
    )

    args = cli._parse_args()

    assert args.command == "run"
    assert args.text_only is True


def test_run_structured_defaults(monkeypatch):
    monkeypatch.delenv("PAPER_OCR_DIGITAL_STRUCTURED", raising=False)
    monkeypatch.delenv("PAPER_OCR_MARKER_COMMAND", raising=False)
    monkeypatch.delenv("PAPER_OCR_GROBID_URL", raising=False)
    monkeypatch.delenv("PAPER_OCR_MARKER_TIMEOUT", raising=False)
    monkeypatch.delenv("PAPER_OCR_GROBID_TIMEOUT", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "run",
            "data/in",
            "out",
        ],
    )
    args = cli._parse_args()
    assert args.digital_structured == "auto"
    assert args.structured_backend == "hybrid"
    assert args.marker_command == "marker_single"
    assert args.marker_url == ""
    assert args.marker_timeout == 120
    assert args.grobid_url == ""
    assert args.grobid_timeout == 60
    assert args.structured_max_workers == 4
    assert args.structured_asset_level == "standard"
    assert args.extract_structured_data is True
    assert args.deplot_command == ""
    assert args.deplot_timeout == 90


def test_run_structured_export_overrides(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "run",
            "data/in",
            "out",
            "--no-extract-structured-data",
            "--deplot-command",
            "deplot-cli --image {image}",
            "--deplot-timeout",
            "30",
        ],
    )
    args = cli._parse_args()
    assert args.extract_structured_data is False
    assert args.deplot_command == "deplot-cli --image {image}"
    assert args.deplot_timeout == 30


def test_run_marker_service_override(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper-ocr",
            "run",
            "data/in",
            "out",
            "--marker-url",
            "http://127.0.0.1:8008",
        ],
    )
    args = cli._parse_args()
    assert args.marker_url == "http://127.0.0.1:8008"


def test_fetch_telegram_requires_env(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(
        doi_csv=tmp_path / "papers.csv",
        output_root=tmp_path / "jobs",
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    args.doi_csv.write_text("DOI\n10.1000/abc\n")

    monkeypatch.delenv("TG_API_ID", raising=False)
    monkeypatch.delenv("TG_API_HASH", raising=False)

    with pytest.raises(SystemExit, match="TG_API_ID"):
        asyncio.run(cli._run_fetch_telegram(args))


def test_fetch_telegram_dispatches(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(
        doi_csv=tmp_path / "papers.csv",
        output_root=tmp_path / "jobs",
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    args.doi_csv.write_text("DOI\n10.1000/abc\n")

    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "abc")
    monkeypatch.setattr(cli, "fetch_from_telegram", _fake_fetch_from_telegram)

    asyncio.run(cli._run_fetch_telegram(args))
    config = _fake_fetch_from_telegram.last_config
    assert config.in_dir == tmp_path / "jobs" / "papers" / "pdfs"
    assert config.report_file == tmp_path / "jobs" / "papers" / "reports" / "telegram_download_report.csv"


def test_fetch_telegram_normalizes_job_slug(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(
        doi_csv=tmp_path / "My.Papers.csv",
        output_root=tmp_path / "jobs",
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    args.doi_csv.write_text("DOI\n10.1000/abc\n")

    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "abc")
    monkeypatch.setattr(cli, "fetch_from_telegram", _fake_fetch_from_telegram)

    asyncio.run(cli._run_fetch_telegram(args))
    config = _fake_fetch_from_telegram.last_config
    assert config.in_dir == tmp_path / "jobs" / "my_papers" / "pdfs"


def test_fetch_telegram_migrates_legacy_default_job_dir(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        doi_csv=tmp_path / "papers.csv",
        output_root=Path("data/jobs"),
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    args.doi_csv.write_text("DOI\n10.1000/abc\n")
    legacy_job = tmp_path / "data" / "telegram_jobs" / "papers"
    (legacy_job / "reports").mkdir(parents=True, exist_ok=True)
    (legacy_job / "reports" / "download_index.json").write_text("{}")

    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "abc")
    monkeypatch.setattr(cli, "fetch_from_telegram", _fake_fetch_from_telegram)

    asyncio.run(cli._run_fetch_telegram(args))

    assert not legacy_job.exists()
    assert (tmp_path / "data" / "jobs" / "papers" / "reports" / "download_index.json").exists()


def test_fetch_telegram_keeps_input_csv_outside_job_folder(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "jobs"
    doi_csv = tmp_path / "papers.csv"
    doi_csv.write_text("DOI\n10.1000/abc\n")
    args = argparse.Namespace(
        doi_csv=doi_csv,
        output_root=output_root,
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "abc")
    monkeypatch.setattr(cli, "fetch_from_telegram", _fake_fetch_from_telegram)

    asyncio.run(cli._run_fetch_telegram(args))
    assert _fake_fetch_from_telegram.last_config.doi_csv == doi_csv
    assert not (output_root / "papers" / "input").exists()


def test_fetch_telegram_does_not_create_ocr_out_subdir(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(
        doi_csv=tmp_path / "papers.csv",
        output_root=tmp_path / "jobs",
        doi_column="DOI",
        target_bot="@example_bot",
        session_name="nexus_session",
        min_delay=10.0,
        max_delay=20.0,
        response_timeout=60,
        search_timeout=40,
        report_file=None,
        failed_file=None,
        debug=False,
    )
    args.doi_csv.write_text("DOI\n10.1000/abc\n")
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "abc")
    monkeypatch.setattr(cli, "fetch_from_telegram", _fake_fetch_from_telegram)

    asyncio.run(cli._run_fetch_telegram(args))
    assert not (tmp_path / "jobs" / "papers" / "ocr_out").exists()
    assert not (tmp_path / "jobs" / "papers" / "input").exists()


def test_run_rejects_output_under_jobs_folder(tmp_path: Path):
    args = argparse.Namespace(
        in_dir=tmp_path / "in",
        out_dir=Path("data/jobs/example/ocr_out"),
    )
    with pytest.raises(SystemExit, match="Refusing to write final OCR outputs under data/jobs"):
        asyncio.run(cli._run(args))


def test_render_dims_for_route_matches_truncation_behavior():
    w, h = cli._render_dims_for_route(610.0, 792.0, "unanchored", max_dim=10000)
    assert w == 2541
    assert h == 3300


def test_final_doc_dir_avoids_collision_on_different_sha(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(out_dir=tmp_path / "out")
    pdf_parent = tmp_path / "in"
    pdf_parent.mkdir()
    pdf_path = pdf_parent / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123def456zzz")
    out = cli._final_doc_dir(args, pdf_path, {"authors": ["Doe, Jane"], "year": "2024", "title": "T"})
    assert out.name == "doc_abc123def456"


def test_final_doc_dir_avoids_non_manifest_existing_folder(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(out_dir=tmp_path / "out")
    pdf_parent = tmp_path / "in"
    pdf_parent.mkdir()
    pdf_path = pdf_parent / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    group_dir = args.out_dir / "in"
    candidate = group_dir / "doc_abc123def456"
    candidate.mkdir(parents=True)
    (candidate / "orphan.txt").write_text("orphan")

    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123def456zzz")

    out = cli._final_doc_dir(args, pdf_path, {"authors": ["Doe, Jane"], "year": "2024", "title": "T"})
    assert out.name == "doc_abc123def456"


def test_process_pdf_skips_when_manifest_sha_matches(monkeypatch, tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    pdf_path = in_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        out_dir=out_dir,
        workers=1,
        model="m",
        base_url="https://example.com/v1/openai",
        max_tokens=32,
        force=False,
        mode="auto",
        debug=False,
        scan_preprocess=False,
        text_only=False,
        metadata_model="meta",
    )
    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123def456zzz")
    doc_dir = out_dir / "in" / "doc_abc123def456"
    (doc_dir / "metadata").mkdir(parents=True)
    (doc_dir / "metadata" / "manifest.json").write_text(
        json.dumps({"sha256": "abc123def456zzz", "page_count": 7, "bibliography": {"title": "T"}})
    )
    (doc_dir / "metadata" / "discovery.json").write_text(json.dumps({"paper_summary": "x", "key_topics": [], "sections": []}))
    monkeypatch.setattr(cli.fitz, "open", lambda p: (_ for _ in ()).throw(AssertionError("fitz.open should not run on skip")))
    result = asyncio.run(cli._process_pdf(args, pdf_path))
    assert result["status"] == "skipped"
    assert result["page_count"] == 7
    assert Path(result["doc_dir"]).name == "doc_abc123def456"


def test_run_raises_when_any_doc_failed(monkeypatch, tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir(parents=True)
    pdf_a = in_dir / "a.pdf"
    pdf_b = in_dir / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    args = argparse.Namespace(in_dir=in_dir, out_dir=out_dir)
    monkeypatch.setattr(cli, "discover_pdfs", lambda p: [pdf_a, pdf_b])

    async def _fake_process(_args, pdf):  # noqa: ANN001
        return {
            "group_dir": str(out_dir / "in"),
            "doc_dir": str(out_dir / "in" / pdf.stem),
            "folder_name": pdf.stem,
            "consolidated_markdown": "document.md",
            "page_count": 1,
            "bibliography": {},
            "discovery": {},
            "status": "failed" if pdf.name == "a.pdf" else "ok",
            "strict_failure": pdf.name == "a.pdf",
            "errors": ["strict"] if pdf.name == "a.pdf" else [],
        }

    monkeypatch.setattr(cli, "_process_pdf", _fake_process)
    monkeypatch.setattr(cli, "_write_group_readmes", lambda records: None)
    with pytest.raises(SystemExit, match="failed_docs"):
        asyncio.run(cli._run(args))


def test_run_export_structured_data_raises_when_strict_doc_fails(monkeypatch, tmp_path: Path):
    root = tmp_path / "out"
    doc_dir = root / "group" / "doc_abc"
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / "metadata").mkdir(parents=True)
    manifest_path = doc_dir / "metadata" / "manifest.json"
    manifest_path.write_text(json.dumps({"doc_id": "abc"}))

    def _fake_build(**kwargs):  # noqa: ANN001
        raise RuntimeError("Strict table artifact mode failed: boom")

    monkeypatch.setattr(cli, "build_structured_exports", _fake_build)
    args = argparse.Namespace(
        ocr_out_dir=root,
        deplot_command="",
        deplot_timeout=30,
        table_source="marker-first",
        table_ocr_merge=True,
        table_ocr_merge_scope="header",
        table_header_ocr_auto=False,
        table_header_ocr_model="m",
        table_header_ocr_max_tokens=500,
        table_artifact_mode="strict",
        ocr_html_dir=None,
        table_quality_gate=True,
        table_escalation="auto",
        table_escalation_max=20,
        table_qa_mode="warn",
        compare_ocr_html=False,
    )
    with pytest.raises(SystemExit, match="failed_docs"):
        cli._run_export_structured_data(args)


def test_ensure_header_ocr_artifacts_generates_only_missing(monkeypatch, tmp_path: Path):
    doc_dir = tmp_path / "doc"
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    table_rows = [
        {"page": 1, "bbox": [10, 10, 100, 100]},
        {"page": 1, "bbox": [110, 10, 200, 100]},
    ]
    (marker_root / "tables_raw.jsonl").write_text("\n".join(json.dumps(r) for r in table_rows) + "\n")
    manifest_path = doc_dir / "metadata" / "manifest.json"
    src_pdf = tmp_path / "source.pdf"
    with fitz.open() as out:
        out.new_page(width=300, height=300)
        out.save(src_pdf)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"source_path": str(src_pdf)}))
    qa_out = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    qa_out.mkdir(parents=True, exist_ok=True)
    (qa_out / "table_01_page_0001.md").write_text("<table></table>")

    class _Resp:
        content = "<table><tr><th>A</th></tr></table>"

    async def _fake_ocr(**kwargs):  # noqa: ANN001
        return _Resp()

    monkeypatch.setattr(cli, "call_olmocr", _fake_ocr)
    summary = asyncio.run(
        cli._ensure_header_ocr_artifacts(
            doc_dir=doc_dir,
            client=object(),  # type: ignore[arg-type]
            model="m",
            max_tokens=100,
        )
    )
    assert summary["expected"] == 2
    assert summary["generated"] == 1
    assert summary["skipped_existing"] == 1
    assert summary["failed"] == 0


def test_process_page_structured_success_skips_fallback(monkeypatch, tmp_path: Path):
    dirs = cli.ensure_dirs(tmp_path / "doc")
    called = {"fallback": False}

    async def _fake_fallback(**kwargs):  # noqa: ANN001
        called["fallback"] = True
        return {}

    monkeypatch.setattr(cli, "_process_page", _fake_fallback)
    monkeypatch.setattr(
        cli,
        "run_marker_page",
        lambda *args, **kwargs: StructuredPageResult(
            success=True,
            markdown="##Intro\n\n-  item",
            artifacts={"markdown": "x"},
        ),
    )

    result = asyncio.run(
        cli._process_page_structured(
            page_index=0,
            pdf_path=tmp_path / "doc.pdf",
            marker_command="marker_single",
            marker_url="",
            marker_timeout=5,
            structured_asset_level="standard",
            structured_backend="hybrid",
            structured_semaphore=asyncio.Semaphore(1),
            route="anchored",
            heuristics=TextHeuristics(
                char_count=1000,
                printable_ratio=0.99,
                cid_ratio=0.0,
                replacement_char_ratio=0.0,
                avg_token_length=5.0,
            ),
            fallback_kwargs={"dirs": dirs, "force": False},
        )
    )

    assert result["status"] == "structured_ok"
    assert not called["fallback"]
    assert Path(result["output_files"]["markdown"]).exists()


def test_process_page_structured_failure_uses_fallback(monkeypatch, tmp_path: Path):
    dirs = cli.ensure_dirs(tmp_path / "doc")

    async def _fake_fallback(**kwargs):  # noqa: ANN001
        return {
            "page_index": kwargs["page_index"],
            "route": kwargs["route_override"],
            "heuristics": {},
            "status": "ok",
            "output_files": {"markdown": str(dirs["pages"] / "0001.md")},
        }

    monkeypatch.setattr(cli, "_process_page", _fake_fallback)
    monkeypatch.setattr(
        cli,
        "run_marker_page",
        lambda *args, **kwargs: StructuredPageResult(success=False, error="marker unavailable"),
    )

    result = asyncio.run(
        cli._process_page_structured(
            page_index=0,
            pdf_path=tmp_path / "doc.pdf",
            marker_command="marker_single",
            marker_url="",
            marker_timeout=5,
            structured_asset_level="standard",
            structured_backend="hybrid",
            structured_semaphore=asyncio.Semaphore(1),
            route="anchored",
            heuristics=TextHeuristics(
                char_count=1000,
                printable_ratio=0.99,
                cid_ratio=0.0,
                replacement_char_ratio=0.0,
                avg_token_length=5.0,
            ),
            fallback_kwargs={"dirs": dirs, "force": False, "page_index": 0},
        )
    )

    assert result["status"] == "structured_fallback"
    assert result["structured"]["fallback_reason"] == "marker unavailable"


def test_extract_discovery_can_return_sections_without_grobid(monkeypatch):
    class _Resp:
        def __init__(self, content: str):
            self.content = content
            self.reasoning_content = ""

    calls: list[str] = []

    async def _fake_text_model(**kwargs):  # noqa: ANN001
        prompt = str(kwargs.get("prompt", ""))
        calls.append(prompt)
        if "extract the paper abstract" in prompt.lower():
            return _Resp('{"abstract":"A study of rheology.","key_topics":["rheology"]}')
        if "one chunk of a paper markdown transcript" in prompt:
            return _Resp(
                '{"chunk_summary":"chunk","key_topics":["rheology"],"sections":[{"title":"Methods","start_page":2,"end_page":4,"summary":"m"}]}'
            )
        if "combining chunk-level paper discovery data" in prompt:
            return _Resp(
                '{"paper_summary":"A study of rheology.","key_topics":["rheology"],"sections":[{"title":"Methods","start_page":2,"end_page":4,"summary":"m"}]}'
            )
        return _Resp("{}")

    monkeypatch.setattr(cli, "call_text_model", _fake_text_model)

    out = asyncio.run(
        cli._extract_discovery(
            client=object(),  # type: ignore[arg-type]
            metadata_model="m",
            bibliography={"title": "T", "citation": "C"},
            page_count=10,
            consolidated_markdown="# Page 1\nAbstract\n\n# Page 2\nMethods\n",
        )
    )

    assert out["paper_summary"]
    assert out["sections"]
    assert any("one chunk of a paper markdown transcript" in p for p in calls)
