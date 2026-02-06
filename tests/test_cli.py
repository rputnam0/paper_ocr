import argparse
import asyncio
from pathlib import Path

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
