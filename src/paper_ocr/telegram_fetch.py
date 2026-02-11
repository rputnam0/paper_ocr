from __future__ import annotations

import asyncio
import csv
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, unquote, urlparse

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
DOI_MATCH_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]+")
LEADING_SYMBOLS_RE = re.compile(r"^[^A-Za-z0-9]+")
DOI_TRAILING_CHARS = ".,;:)]}>\"'"

FAILURE_MARKERS = (
    "not found",
    "unavailable",
    "insufficient point",
    "points exhausted",
    "no points",
)
SEARCHING_MARKERS = ("searching", "processing", "queued", "looking at")
UPLOAD_PROGRESS_MARKERS = ("uploaded to telegram", "uploading to telegram")
POINTS_LABEL_RE = re.compile(r"(\d+)\s*pts?", re.IGNORECASE)

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
MAX_FILENAME_STEM = 180
MAX_DOI_REQUERY_ATTEMPTS = 3


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
    target_bot: str = ""
    session_name: str = "nexus_session"
    min_delay: float = 10.0
    max_delay: float = 20.0
    response_timeout: int = 15
    search_timeout: int = 40
    report_file: Path | None = None
    failed_file: Path | None = None
    debug: bool = False


def normalize_doi(raw: str) -> str:
    resolved = resolve_doi_from_raw(raw)
    if resolved:
        return resolved
    doi = DOI_PREFIX_RE.sub("", (raw or "").strip())
    return doi.lower()


def _clean_doi_candidate(raw: str) -> str:
    candidate = DOI_PREFIX_RE.sub("", (raw or "").strip())
    candidate = candidate.strip().strip("<>{}[]()")
    while candidate and candidate[-1] in DOI_TRAILING_CHARS:
        candidate = candidate[:-1]
    return candidate.strip().lower()


def _looks_like_doi(value: str) -> bool:
    return bool(DOI_MATCH_RE.fullmatch(value))


def _extract_dois_from_text(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _record(value: str) -> None:
        cleaned = _clean_doi_candidate(value)
        if cleaned and _looks_like_doi(cleaned) and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)

    text = (raw or "").strip()
    if not text:
        return out

    _record(text)

    decoded = unquote(text)
    for source in {text, decoded}:
        for match in DOI_MATCH_RE.findall(source):
            _record(match)
        parsed = urlparse(source)
        if parsed.scheme and parsed.netloc:
            for value in parse_qs(parsed.query).get("doi", []):
                _record(value)
            if "doi.org" in parsed.netloc.lower() and parsed.path:
                _record(parsed.path.lstrip("/"))
    return out


def resolve_doi_from_raw(raw: str) -> str:
    candidates = _extract_dois_from_text(raw)
    return candidates[0] if candidates else ""


def doi_filename(doi: str) -> str:
    replaced = doi.replace("/", "_")
    safe = SAFE_CHAR_RE.sub("_", replaced).strip("_")
    stem = (safe or "unknown_doi")[:MAX_FILENAME_STEM].rstrip("._-")
    return f"{stem or 'unknown_doi'}.pdf"


def extract_bot_title(text: str) -> str:
    if not text.strip():
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    cleaned = LEADING_SYMBOLS_RE.sub("", first_line).strip()
    return cleaned


def title_filename(title: str, doi: str) -> str:
    if not title.strip():
        return doi_filename(doi)
    safe = SAFE_CHAR_RE.sub("_", title).strip("_")
    stem = (safe or doi_filename(doi).removesuffix(".pdf"))[:MAX_FILENAME_STEM].rstrip("._-")
    return f"{stem or doi_filename(doi).removesuffix('.pdf')}.pdf"


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


def load_download_index(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        import json

        raw = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def write_download_index(path: Path, index: dict[str, str]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=True))


def load_index_from_report(report_file: Path, in_dir: Path) -> dict[str, str]:
    if not report_file.exists():
        return {}
    out: dict[str, str] = {}
    try:
        with report_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = (row.get("status", "") or "").strip()
                doi = (row.get("doi_normalized", "") or "").strip().lower()
                file_path = (row.get("file_path", "") or "").strip()
                if status not in {"Success", "Exists"} or not doi or not file_path:
                    continue
                p = Path(file_path)
                if not p.exists():
                    continue
                try:
                    out[doi] = str(p.relative_to(in_dir))
                except Exception:
                    out[doi] = p.name
    except Exception:
        return {}
    return out


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


def _button_urls(message: Any) -> list[str]:
    buttons = getattr(message, "buttons", None)
    if not buttons:
        return []
    urls: list[str] = []
    for row in buttons:
        for button in row:
            url = getattr(button, "url", None)
            if isinstance(url, str) and url.strip():
                urls.append(url.strip())
    return urls


def _link_button_index(message: Any) -> int | None:
    buttons = getattr(message, "buttons", None)
    if not buttons:
        return None
    idx = 0
    for row in buttons:
        for button in row:
            label = str(getattr(button, "text", "") or "")
            url = getattr(button, "url", None)
            label_low = label.lower()
            if isinstance(url, str) and url.strip():
                return idx
            if "🔗" in label or "link" in label_low:
                return idx
            idx += 1
    return None


def _message_doi_candidates(message: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    raw_values = [_message_text(message), *_button_urls(message)]
    for raw in raw_values:
        for doi in _extract_dois_from_text(raw):
            if doi not in seen:
                seen.add(doi)
                out.append(doi)
    return out


def _request_button_index(message: Any) -> int | None:
    labels = _button_labels(message)
    for idx, label in enumerate(labels):
        if "request" in label.lower():
            return idx
    return None


def _pdf_button_index(message: Any) -> int | None:
    labels = _button_labels(message)
    for idx, label in enumerate(labels):
        if "pdf" in label.lower():
            return idx
    return None


def _confirm_button_index(message: Any) -> int | None:
    labels = _button_labels(message)
    for idx, label in enumerate(labels):
        low = label.lower()
        if "confirm" in low or "✅" in label:
            return idx
    return None


def _required_points(message: Any) -> int | None:
    labels = _button_labels(message)
    for label in labels:
        match = POINTS_LABEL_RE.search(label)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _is_hard_failure(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in FAILURE_MARKERS)


def _is_searching(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in SEARCHING_MARKERS) or any(
        marker in low for marker in UPLOAD_PROGRESS_MARKERS
    )


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
    existing_file_path: Path | None = None,
    debug: bool = False,
) -> FetchResult:
    started = _utc_now()
    fallback_path = in_dir / doi_filename(doi_normalized)

    if existing_file_path is not None and existing_file_path.exists():
        finished = _utc_now()
        return FetchResult(
            doi_original=doi_original,
            doi_normalized=doi_normalized,
            status="Exists",
            file_path=str(existing_file_path),
            error="",
            bot_message_excerpt="",
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            elapsed_s=(finished - started).total_seconds(),
        )

    if fallback_path.exists():
        finished = _utc_now()
        return FetchResult(
            doi_original=doi_original,
            doi_normalized=doi_normalized,
            status="Exists",
            file_path=str(fallback_path),
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
    attempted_queries: set[str] = {doi_normalized}
    requery_attempts = 0
    clicked_link_buttons: set[tuple[int, int]] = set()

    async def _next_response(conv: Any, *, edit_from: Any | None = None) -> Any:
        while True:
            try:
                try:
                    return await _with_floodwait(conv.get_response)
                except asyncio.TimeoutError:
                    if edit_from is not None and hasattr(conv, "get_edit"):
                        return await _with_floodwait(lambda: conv.get_edit(message=edit_from))
                    raise
            except asyncio.TimeoutError:
                if saw_searching and search_deadline is not None and time.monotonic() < search_deadline:
                    continue
                raise

    try:
        async with conversation_factory() as conv:
            sent_message = await _with_floodwait(lambda: conv.send_message(doi_normalized))
            response = await _next_response(conv)

            while True:
                text = _message_text(response)
                excerpt = _excerpt(text)
                if debug:
                    labels = _button_labels(response)
                    print(
                        f"[telegram-fetch] doi={doi_normalized} "
                        f"file={bool(getattr(response, 'file', None))} "
                        f"text={excerpt!r} buttons={labels}"
                    )

                if getattr(response, "file", None):
                    title = extract_bot_title(text)
                    candidate = in_dir / title_filename(title, doi_normalized)
                    if candidate.exists() and candidate != fallback_path:
                        candidate = in_dir / f"{candidate.stem}_{doi_filename(doi_normalized).removesuffix('.pdf')}.pdf"
                    await _with_floodwait(lambda: response.download_media(file=str(candidate)))
                    status = "Success"
                    fallback_path = candidate
                    break

                button_idx = _request_button_index(response)
                if button_idx is not None:
                    await _with_floodwait(lambda: response.click(button_idx))
                    response = await _next_response(conv, edit_from=sent_message)
                    continue

                pdf_button_idx = _pdf_button_index(response)
                if pdf_button_idx is not None:
                    await _with_floodwait(lambda: response.click(pdf_button_idx))
                    response = await _next_response(conv)
                    continue

                confirm_idx = _confirm_button_index(response)
                if confirm_idx is not None:
                    required_points = _required_points(response)
                    if required_points is not None and required_points <= 0:
                        await _with_floodwait(lambda: response.click(confirm_idx))
                        saw_searching = True
                        if search_deadline is None:
                            search_deadline = time.monotonic() + max(search_timeout, response_timeout)
                        response = await _next_response(conv, edit_from=sent_message)
                        continue

                if _is_hard_failure(text):
                    status = "Failed"
                    error = text or "Not found"
                    break

                if _is_searching(text):
                    saw_searching = True
                    if search_deadline is None:
                        search_deadline = time.monotonic() + max(search_timeout, response_timeout)
                    response = await _next_response(conv, edit_from=sent_message)
                    continue

                doi_candidates = _message_doi_candidates(response)
                if requery_attempts < MAX_DOI_REQUERY_ATTEMPTS:
                    next_doi = next((doi for doi in doi_candidates if doi not in attempted_queries), "")
                    if next_doi:
                        attempted_queries.add(next_doi)
                        requery_attempts += 1
                        sent_message = await _with_floodwait(lambda: conv.send_message(next_doi))
                        response = await _next_response(conv)
                        continue

                link_button_idx = _link_button_index(response)
                response_key = int(getattr(response, "id", 0) or id(response))
                if (
                    link_button_idx is not None
                    and (response_key, link_button_idx) not in clicked_link_buttons
                ):
                    clicked_link_buttons.add((response_key, link_button_idx))
                    await _with_floodwait(lambda: response.click(link_button_idx))
                    response = await _next_response(conv, edit_from=sent_message)
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
        file_path=str(fallback_path) if status in {"Success", "Exists"} else "",
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
    index_file = report_file.parent / "download_index.json"

    if config.min_delay < 0 or config.max_delay < 0 or config.min_delay > config.max_delay:
        raise ValueError("Invalid delay bounds: require 0 <= min_delay <= max_delay")

    dois = load_unique_dois(config.doi_csv, config.doi_column)
    if not dois:
        write_reports([], report_file, failed_file)
        return []

    rows: list[FetchResult] = []
    index = load_download_index(index_file)
    index.update(load_index_from_report(report_file, config.in_dir))

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
                existing_file_path=(config.in_dir / index[doi_normalized]) if doi_normalized in index else None,
                debug=config.debug,
            )
            rows.append(result)
            if result.status in {"Success", "Exists"} and result.file_path:
                p = Path(result.file_path)
                try:
                    rel = p.relative_to(config.in_dir)
                    index[doi_normalized] = str(rel)
                except Exception:
                    index[doi_normalized] = p.name

            if result.status != "Exists":
                await asyncio.sleep(random.uniform(config.min_delay, config.max_delay))

            # Persist progress incrementally so interrupted runs can resume.
            write_reports(rows, report_file, failed_file)
            write_download_index(index_file, index)
    finally:
        await client.disconnect()

    write_reports(rows, report_file, failed_file)
    write_download_index(index_file, index)
    return rows
