import asyncio
import csv
from pathlib import Path

from paper_ocr import telegram_fetch


class _FakeFloodWaitError(Exception):
    def __init__(self, seconds: int):
        super().__init__(f"wait {seconds}")
        self.seconds = seconds


class _FakeButton:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str = "", has_file: bool = False, buttons=None):
        self.text = text
        self.raw_text = text
        self.file = object() if has_file else None
        self.buttons = buttons
        self.clicked = []
        self.downloaded_to = None

    async def click(self, index: int):
        self.clicked.append(index)

    async def download_media(self, file: str):
        self.downloaded_to = file
        Path(file).write_bytes(b"pdf")


class _FakeConversation:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def send_message(self, text: str):
        self.sent.append(text)

    async def get_response(self):
        if not self._responses:
            raise AssertionError("no more responses")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _ConversationFactory:
    def __init__(self, conv):
        self.conv = conv

    async def __aenter__(self):
        return self.conv

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _factory(conv):
    return lambda: _ConversationFactory(conv)


def test_normalize_doi():
    assert telegram_fetch.normalize_doi(" https://doi.org/10.1000/ABC ") == "10.1000/abc"
    assert telegram_fetch.normalize_doi("DOI:10.5555/xyz") == "10.5555/xyz"


def test_doi_filename_sanitizes():
    assert telegram_fetch.doi_filename("10.1038/nature123") == "10.1038_nature123.pdf"
    assert telegram_fetch.doi_filename("10.1000/a:b*c") == "10.1000_a_b_c.pdf"


def test_load_unique_dois_dedups(tmp_path: Path):
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text("DOI\n10.1000/abc\nhttps://doi.org/10.1000/ABC\n\n")

    dois = telegram_fetch.load_unique_dois(csv_path, "DOI")

    assert dois == [("10.1000/abc", "10.1000/abc")]


def test_process_doi_exists_skips_network(tmp_path: Path):
    path = tmp_path / "10.1000_abc.pdf"
    path.write_bytes(b"pdf")

    async def _unused():
        raise AssertionError("conversation should not be created")

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_unused,
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Exists"


def test_process_doi_instant_file_success(tmp_path: Path):
    msg = _FakeMessage(has_file=True)
    conv = _FakeConversation([msg])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Success"
    assert conv.sent == ["10.1000/abc"]
    assert (tmp_path / "10.1000_abc.pdf").exists()


def test_process_doi_cache_miss_with_request_button(tmp_path: Path):
    cache_miss = _FakeMessage(
        text="DOI not found in cache",
        buttons=[[ _FakeButton("Request (1 point)") ]],
    )
    file_msg = _FakeMessage(has_file=True)
    conv = _FakeConversation([cache_miss, file_msg])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Success"
    assert cache_miss.clicked == [0]


def test_process_doi_hard_failure(tmp_path: Path):
    conv = _FakeConversation([_FakeMessage(text="Not found")])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Failed"


def test_process_doi_unknown_response(tmp_path: Path):
    conv = _FakeConversation([_FakeMessage(text="Working on it")])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "UnknownResponse"


def test_process_doi_timeout(tmp_path: Path):
    conv = _FakeConversation([asyncio.TimeoutError()])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Timeout"


def test_process_doi_floodwait_then_success(tmp_path: Path, monkeypatch):
    sleeps = []

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(telegram_fetch.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(telegram_fetch, "FloodWaitError", _FakeFloodWaitError)

    conv = _FakeConversation([_FakeFloodWaitError(3), _FakeMessage(has_file=True)])

    result = asyncio.run(
        telegram_fetch.process_doi(
            conversation_factory=_factory(conv),
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            in_dir=tmp_path,
        )
    )

    assert result.status == "Success"
    assert sleeps == [3]


def test_write_reports(tmp_path: Path):
    rows = [
        telegram_fetch.FetchResult(
            doi_original="10.1000/abc",
            doi_normalized="10.1000/abc",
            status="Success",
            file_path="/tmp/a.pdf",
            error="",
            bot_message_excerpt="",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
        ),
        telegram_fetch.FetchResult(
            doi_original="10.1000/missing",
            doi_normalized="10.1000/missing",
            status="Failed",
            file_path="",
            error="Not found",
            bot_message_excerpt="Not found",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
            elapsed_s=1.0,
        ),
    ]

    report = tmp_path / "report.csv"
    failed = tmp_path / "failed.csv"
    telegram_fetch.write_reports(rows, report, failed)

    with report.open(newline="") as f:
        report_rows = list(csv.DictReader(f))
    with failed.open(newline="") as f:
        failed_rows = list(csv.DictReader(f))

    assert report_rows[0]["status"] == "Success"
    assert len(failed_rows) == 1
    assert failed_rows[0]["status"] == "Failed"
