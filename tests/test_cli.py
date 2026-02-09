import argparse
import asyncio
from pathlib import Path

import pytest

from paper_ocr import cli


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


def test_fetch_telegram_requires_env(monkeypatch, tmp_path: Path):
    args = argparse.Namespace(
        doi_csv=tmp_path / "papers.csv",
        output_root=tmp_path / "jobs",
        doi_column="DOI",
        target_bot="@your_bot_username",
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
        target_bot="@your_bot_username",
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
