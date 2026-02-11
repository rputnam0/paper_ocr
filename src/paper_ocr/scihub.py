from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_SCIHUB_INDEX_URL = "https://sci-hub.now.sh/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.IGNORECASE)
_IFRAME_SRC_RE = re.compile(r"<iframe[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _normalize_base_url(url: str) -> str:
    out = (url or "").strip()
    if not out:
        return ""
    if not out.startswith(("http://", "https://")):
        out = f"https://{out.lstrip('/')}"
    return out.rstrip("/")


def parse_scihub_base_urls(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []

    values: Iterable[str]
    if isinstance(raw, str):
        values = re.split(r"[\s,]+", raw)
    else:
        values = raw

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_base_url(str(value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _http_get(url: str, timeout: int) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
        content_type = str(response.headers.get("Content-Type", ""))
    return data, content_type


def discover_scihub_base_urls(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> list[str]:
    try:
        html, _ = _http_get(DEFAULT_SCIHUB_INDEX_URL, timeout)
    except Exception:
        return []

    decoded = html.decode("utf-8", errors="ignore")
    urls: list[str] = []
    seen: set[str] = set()
    for href in _HREF_RE.findall(decoded):
        candidate = _normalize_base_url(href)
        if "sci-hub" not in candidate.lower() or candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return urls


def _extract_iframe_src(html: bytes) -> str:
    decoded = html.decode("utf-8", errors="ignore")
    match = _IFRAME_SRC_RE.search(decoded)
    if not match:
        return ""
    return (match.group(1) or "").strip()


def _resolve_pdf_url(identifier: str, base_url: str, timeout: int) -> str:
    safe_identifier = quote(identifier.strip(), safe="/:._-()")
    lookup_url = f"{base_url}/{safe_identifier}"
    page_bytes, page_content_type = _http_get(lookup_url, timeout)

    if "application/pdf" in page_content_type.lower():
        return lookup_url

    iframe_src = _extract_iframe_src(page_bytes)
    if not iframe_src:
        return ""
    if iframe_src.startswith("//"):
        return f"https:{iframe_src}"
    return urljoin(f"{base_url}/", iframe_src)


def download_pdf_via_scihub(
    *,
    identifier: str,
    output_path: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    base_urls: Sequence[str] | None = None,
) -> Path | None:
    normalized_identifier = (identifier or "").strip()
    if not normalized_identifier:
        return None

    candidates = parse_scihub_base_urls(base_urls)
    if not candidates:
        candidates = discover_scihub_base_urls(timeout=timeout)
    if not candidates:
        return None

    for base_url in candidates:
        try:
            pdf_url = _resolve_pdf_url(normalized_identifier, base_url, timeout)
            if not pdf_url:
                continue
            pdf_bytes, content_type = _http_get(pdf_url, timeout)
            looks_like_pdf = "application/pdf" in content_type.lower() or pdf_bytes.startswith(b"%PDF")
            if not looks_like_pdf:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)
            return output_path
        except Exception:
            continue
    return None
