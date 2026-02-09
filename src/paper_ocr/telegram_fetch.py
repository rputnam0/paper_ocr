from __future__ import annotations

import asyncio
import csv
import random
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError
except Exception:  # pragma: no cover - exercised only when dependency missing
    TelegramClient = Any  # type: ignore[assignment,misc]

    class FloodWaitError(Exception):
        seconds: int = 0

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - exercised only when dependency missing
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]+")

FAILURE_MARKERS = (
    "not found",
    "unavailable",
    "insufficient point",
    "points exhausted",
    "no points",
)
SEARCHING_MARKERS = ("searching", "processing", "queued")

REPORT_COLUMNS = [
    "doi_original",
    "doi_normalized",
    "status",
    "file_path",
    "error",
    "bot_message_excerpt",
    "started_at",
    "finished_at",
    "elapsed_s",
]


@dataclass
class FetchResult:
    doi_original: str
    doi_normalized: str
    status: str
    file_path: str
    error: str
    bot_message_excerpt: str
    started_at: str
    finished_at: str
    elapsed_s: float


@dataclass
class FetchTelegramConfig:
    api_id: int
    api_hash: str
    doi_csv: Path
    in_dir: Path
    doi_column: str = "DOI"
    target_bot: str = "@your_bot_username"
    session_name: str = "nexus_session"
    min_delay: float = 10.0
    max_delay: float = 20.0
    response_timeout: int = 15
    search_timeout: int = 40
    report_file: Path | None = None
    failed_file: Path | None = None


def normalize_doi(raw: str) -> str:
    doi = DOI_PREFIX_RE.sub("", (raw or "").strip())
    return doi.lower()


def doi_filename(doi: str) -> str:
    replaced = doi.replace("/", "_")
    safe = SAFE_CHAR_RE.sub("_", replaced).strip("_")
    return f"{safe or 'unknown_doi'}.pdf"


def load_unique_dois(csv_path: Path, doi_column: str) -> list[tuple[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or doi_column not in reader.fieldnames:
            raise ValueError(f"Missing DOI column '{doi_column}' in {csv_path}")

        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in reader:
            original = (row.get(doi_column, "") or "").strip()
            normalized = normalize_doi(original)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append((original, normalized))
        return out


def write_reports(rows: list[FetchResult], report_file: Path, failed_file: Path) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    failed_file.parent.mkdir(parents=True, exist_ok=True)

    with report_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with failed_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            if row.status not in {"Success", "Exists"}:
                writer.writerow(asdict(row))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _excerpt(text: str, max_len: int = 180) -> str:
    squashed = " ".join((text or "").split())
    return squashed[:max_len]


def _message_text(message: Any) -> str:
    return str(getattr(message, "raw_text", None) or getattr(message, "text", "") or "")


def _button_labels(message: Any) -> list[str]:
    buttons = getattr(message, "buttons", None)
    if not buttons:
        return []
    labels: list[str] = []
    for row in buttons:
        for button in row:
            labels.append(str(getattr(button, "text", "") or ""))
    return labels


def _request_button_index(message: Any) -> int | None:
    labels = _button_labels(message)
    for idx, label in enumerate(labels):
        if "request" in label.lower():
            return idx
    return None


def _is_hard_failure(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in FAILURE_MARKERS)


def _is_searching(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in SEARCHING_MARKERS)


async def _with_floodwait(call: Callable[[], Awaitable[Any]]) -> Any:
    while True:
        try:
            return await call()
        except FloodWaitError as exc:
            wait_seconds = max(int(getattr(exc, "seconds", 0)), 0)
            await asyncio.sleep(wait_seconds)


async def process_doi(
    conversation_factory: Callable[[], Any],
    doi_original: str,
    doi_normalized: str,
    in_dir: Path,
    response_timeout: int = 15,
    search_timeout: int = 40,
) -> FetchResult:
    started = _utc_now()
    file_path = in_dir / doi_filename(doi_normalized)

    if file_path.exists():
        finished = _utc_now()
        return FetchResult(
            doi_original=doi_original,
            doi_normalized=doi_normalized,
            status="Exists",
            file_path=str(file_path),
            error="",
            bot_message_excerpt="",
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            elapsed_s=(finished - started).total_seconds(),
        )

    status = "UnknownResponse"
    error = ""
    excerpt = ""
    saw_searching = False
    search_deadline: float | None = None

    async def _next_response(conv: Any) -> Any:
        while True:
            try:
                return await _with_floodwait(conv.get_response)
            except asyncio.TimeoutError:
                if saw_searching and search_deadline is not None and time.monotonic() < search_deadline:
                    continue
                raise

    try:
        async with conversation_factory() as conv:
            await _with_floodwait(lambda: conv.send_message(doi_normalized))
            response = await _next_response(conv)

            while True:
                text = _message_text(response)
                excerpt = _excerpt(text)

                if getattr(response, "file", None):
                    await _with_floodwait(lambda: response.download_media(file=str(file_path)))
                    status = "Success"
                    break

                button_idx = _request_button_index(response)
                if button_idx is not None:
                    await _with_floodwait(lambda: response.click(button_idx))
                    response = await _with_floodwait(conv.get_response)
                    continue

                if _is_hard_failure(text):
                    status = "Failed"
                    error = text or "Not found"
                    break

                if _is_searching(text):
                    saw_searching = True
                    if search_deadline is None:
                        search_deadline = time.monotonic() + max(search_timeout, response_timeout)
                    response = await _next_response(conv)
                    continue

                labels = _button_labels(response)
                status = "UnknownResponse"
                if labels:
                    error = f"Unhandled buttons: {', '.join(labels)}"
                else:
                    error = text or "Unknown bot response"
                break

    except asyncio.TimeoutError:
        status = "Timeout"
        error = "Timed out waiting for bot response"
    except Exception as exc:  # pragma: no cover - defensive guard
        status = "Error"
        error = str(exc)

    finished = _utc_now()
    return FetchResult(
        doi_original=doi_original,
        doi_normalized=doi_normalized,
        status=status,
        file_path=str(file_path) if status in {"Success", "Exists"} else "",
        error=error,
        bot_message_excerpt=excerpt,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        elapsed_s=(finished - started).total_seconds(),
    )


async def fetch_from_telegram(config: FetchTelegramConfig) -> list[FetchResult]:
    config.in_dir.mkdir(parents=True, exist_ok=True)
    report_file = config.report_file or (config.in_dir / "telegram_download_report.csv")
    failed_file = config.failed_file or (config.in_dir / "telegram_failed_papers.csv")

    if config.min_delay < 0 or config.max_delay < 0 or config.min_delay > config.max_delay:
        raise ValueError("Invalid delay bounds: require 0 <= min_delay <= max_delay")

    dois = load_unique_dois(config.doi_csv, config.doi_column)
    if not dois:
        write_reports([], report_file, failed_file)
        return []

    rows: list[FetchResult] = []

    client = TelegramClient(config.session_name, config.api_id, config.api_hash)
    await client.start()
    try:
        await client.get_input_entity(config.target_bot)

        progress = tqdm(dois, desc="Fetching DOIs")
        for doi_original, doi_normalized in progress:
            if hasattr(progress, "set_description"):
                progress.set_description(f"DOI {doi_normalized[:28]}")

            result = await process_doi(
                conversation_factory=lambda: client.conversation(config.target_bot, timeout=config.response_timeout),
                doi_original=doi_original,
                doi_normalized=doi_normalized,
                in_dir=config.in_dir,
                response_timeout=config.response_timeout,
                search_timeout=config.search_timeout,
            )
            rows.append(result)

            if result.status != "Exists":
                await asyncio.sleep(random.uniform(config.min_delay, config.max_delay))
    finally:
        await client.disconnect()

    write_reports(rows, report_file, failed_file)
    return rows
