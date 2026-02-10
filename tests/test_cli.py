import argparse
import asyncio
import json
from pathlib import Path

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


def test_run_export_structured_data_updates_manifest(monkeypatch, tmp_path: Path):
    root = tmp_path / "out"
    doc_dir = root / "group" / "Doe_2024"
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / "metadata").mkdir(parents=True)
    manifest_path = doc_dir / "metadata" / "manifest.json"
    manifest_path.write_text(json.dumps({"doc_id": "abc"}))

    monkeypatch.setattr(
        cli,
        "build_structured_exports",
        lambda **kwargs: StructuredExportSummary(
            table_count=2,
            figure_count=3,
            deplot_count=1,
            unresolved_figure_count=0,
            errors=[],
        ),
    )

    args = argparse.Namespace(
        ocr_out_dir=root,
        deplot_command="deplot-cli --image {image}",
        deplot_timeout=30,
    )
    result = cli._run_export_structured_data(args)
    assert result["docs_processed"] == 1
    payload = json.loads(manifest_path.read_text())
    assert payload["structured_data_extraction"]["table_count"] == 2
    assert payload["structured_data_extraction"]["figure_count"] == 3


def test_run_export_structured_data_requires_docs(tmp_path: Path):
    args = argparse.Namespace(
        ocr_out_dir=tmp_path / "empty_out",
        deplot_command="",
        deplot_timeout=90,
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


def test_fetch_telegram_skips_copy_when_csv_already_in_job_input(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "jobs"
    doi_csv = output_root / "papers" / "input" / "papers.csv"
    doi_csv.parent.mkdir(parents=True, exist_ok=True)
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


def test_final_doc_dir_avoids_collision_on_different_sha(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(out_dir=tmp_path / "out")
    pdf_parent = tmp_path / "in"
    pdf_parent.mkdir()
    pdf_path = pdf_parent / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    bibliography = {"authors": ["Doe, Jane"], "year": "2024", "title": "T"}

    group_dir = args.out_dir / "in"
    candidate = group_dir / "Doe_Jane_2024"
    (candidate / "metadata").mkdir(parents=True)
    (candidate / "metadata" / "manifest.json").write_text(json.dumps({"sha256": "different"}))

    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123def456zzz")

    out = cli._final_doc_dir(args, pdf_path, bibliography)

    assert out.name == "Doe_Jane_2024_abc123def456"


def test_final_doc_dir_avoids_non_manifest_existing_folder(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(out_dir=tmp_path / "out")
    pdf_parent = tmp_path / "in"
    pdf_parent.mkdir()
    pdf_path = pdf_parent / "paper.pdf"
    pdf_path.write_bytes(b"pdf")
    bibliography = {"authors": ["Doe, Jane"], "year": "2024", "title": "T"}

    group_dir = args.out_dir / "in"
    candidate = group_dir / "Doe_Jane_2024"
    candidate.mkdir(parents=True)
    (candidate / "orphan.txt").write_text("orphan")

    monkeypatch.setattr(cli, "file_sha256", lambda p: "abc123def456zzz")

    out = cli._final_doc_dir(args, pdf_path, bibliography)

    assert out.name == "Doe_Jane_2024_abc123def456"


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
