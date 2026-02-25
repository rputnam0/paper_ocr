from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Any, Callable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
DOI_MATCH_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
DOI_TRAILING_CHARS = ".,;:)]}>\"'"
YEAR_RE = re.compile(r"(19|20)\d{2}")
META_TAG_RE = re.compile(r"<meta\s+[^>]*>", re.IGNORECASE)
ATTR_RE = re.compile(r"([A-Za-z0-9_:\-.]+)\s*=\s*['\"]([^'\"]*)['\"]")
JSONLD_RE = re.compile(
    r"<script[^>]*type=['\"]application/ld\+json['\"][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)

INPUT_ALIASES = {
    "doi": ("doi", "DOI", "doi_raw"),
    "url": ("url", "url_landing", "landing_url", "link"),
    "title": ("title", "paper_title"),
    "author": ("first_author", "authors_first_last", "author", "authors"),
    "year": ("year", "publication_year"),
    "container": ("journal_or_repository", "container_title", "journal"),
}

ACCEPTED_FETCH_STATUSES = {
    "canonicalized_crossref",
    "canonicalized_non_crossref",
    "inferred_crossref",
}

WORK_SELECT_FIELDS = "DOI,title,author,issued,type,container-title,URL,score"
SEARCH_SELECT_FIELDS = "DOI,title,author,issued,type,container-title,URL,score"
UNKNOWN_MARKERS = {"unknown", "n/a", "na", "none", "null", "-"}

RESOLVED_COLUMNS = [
    "row_index",
    "doi_raw",
    "url_raw",
    "title",
    "first_author",
    "year",
    "container_title",
    "doi_extracted",
    "doi_canonical",
    "doi_agency",
    "doi_status",
    "doi_match_method",
    "doi_confidence",
    "doi_match_notes",
    "crossref_candidate_dois",
]


@dataclass
class DoiResolutionConfig:
    input_csv: Path
    output_dir: Path
    doi_column: str | None = None
    url_column: str | None = None
    title_column: str | None = None
    author_column: str | None = None
    year_column: str | None = None
    container_column: str | None = None
    crossref_mailto: str = ""
    rows: int = 5
    timeout: int = 20
    max_retries: int = 3
    use_cache: bool = True
    refresh_cache: bool = False
    urlopen: Callable[..., Any] = request.urlopen
    user_agent: str = "paper-ocr/0.1 (+https://github.com)"


@dataclass
class _HttpResult:
    method: str
    url: str
    status: int
    body: str
    headers: dict[str, str]
    from_cache: bool = False
    error: str = ""


@dataclass
class _ResolverState:
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_logs: list[dict[str, Any]] = field(default_factory=list)


def normalize_doi(raw: str) -> str:
    doi = DOI_PREFIX_RE.sub("", (raw or "").strip())
    doi = doi.strip().strip("<>{}[]()")
    while doi and doi[-1] in DOI_TRAILING_CHARS:
        doi = doi[:-1]
    return doi.strip().lower()


def extract_doi_candidates(*values: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _record(candidate: str) -> None:
        doi = normalize_doi(candidate)
        if doi and DOI_MATCH_RE.fullmatch(doi) and doi not in seen:
            seen.add(doi)
            out.append(doi)

    for raw in values:
        text = (raw or "").strip()
        if not text:
            continue
        for source in {text, unquote(text)}:
            _record(source)
            for match in DOI_MATCH_RE.findall(source):
                _record(match)
            parsed = urlparse(source)
            if parsed.scheme and parsed.netloc:
                for qv in parse_qs(parsed.query).get("doi", []):
                    _record(qv)
                if "doi.org" in parsed.netloc.lower() and parsed.path:
                    _record(parsed.path.lstrip("/"))
    return out


def parse_landing_metadata(html_text: str) -> dict[str, str]:
    result = {"title": "", "first_author": "", "year": "", "container_title": ""}
    if not html_text:
        return result

    tags: list[dict[str, str]] = []
    for tag in META_TAG_RE.findall(html_text):
        attrs: dict[str, str] = {}
        for key, value in ATTR_RE.findall(tag):
            attrs[key.lower()] = unescape(value).strip()
        tags.append(attrs)

    def _meta_value(*names: str) -> str:
        names_low = {n.lower() for n in names}
        for attrs in tags:
            key = attrs.get("name", "") or attrs.get("property", "")
            if key.lower() in names_low:
                return attrs.get("content", "").strip()
        return ""

    highwire_title = _meta_value("citation_title")
    highwire_author = _meta_value("citation_author")
    highwire_date = _meta_value("citation_publication_date", "citation_date")
    dc_title = _meta_value("dc.title")
    dc_author = _meta_value("dc.creator")
    dc_date = _meta_value("dc.date")

    result["title"] = highwire_title or dc_title
    result["first_author"] = highwire_author or dc_author
    result["year"] = _extract_year(highwire_date) or _extract_year(dc_date)

    for payload in JSONLD_RE.findall(html_text):
        try:
            parsed = json.loads(unescape(payload).strip())
        except Exception:
            continue
        for obj in _as_list(parsed):
            if not isinstance(obj, dict):
                continue
            if not result["title"]:
                result["title"] = str(obj.get("name") or obj.get("headline") or "").strip()
            if not result["first_author"]:
                result["first_author"] = _jsonld_first_author(obj.get("author"))
            if not result["year"]:
                result["year"] = _extract_year(str(obj.get("datePublished", "")))
            if not result["container_title"]:
                is_part_of = obj.get("isPartOf")
                if isinstance(is_part_of, dict):
                    result["container_title"] = str(is_part_of.get("name", "")).strip()
                elif isinstance(is_part_of, str):
                    result["container_title"] = is_part_of.strip()
    return result


def resolve_dois(config: DoiResolutionConfig) -> dict[str, Any]:
    if not config.input_csv.exists():
        raise ValueError(f"Input CSV does not exist: {config.input_csv}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = config.output_dir / "cache.json"
    state = _ResolverState(cache=_load_cache(cache_path, config), raw_logs=[])

    with config.input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        resolved_columns = _resolve_columns(fieldnames, config)
        input_rows = list(reader)

    resolved_rows: list[dict[str, str]] = []
    candidates_rows: list[dict[str, Any]] = []
    fetch_ready: list[str] = []
    fetch_seen: set[str] = set()
    status_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    agency_counts: dict[str, int] = {}

    for idx, row in enumerate(input_rows, start=1):
        norm = _normalize_row(row, idx, resolved_columns)
        resolved = _resolve_row(norm, config, state)
        resolved_rows.append(resolved["row"])

        status = resolved["row"]["doi_status"]
        method = resolved["row"]["doi_match_method"]
        agency = resolved["row"]["doi_agency"] or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1
        agency_counts[agency] = agency_counts.get(agency, 0) + 1

        for candidate in resolved["candidates"]:
            record = {"row_index": idx, **candidate}
            candidates_rows.append(record)

        canonical = resolved["row"]["doi_canonical"]
        if status in ACCEPTED_FETCH_STATUSES and canonical and canonical not in fetch_seen:
            fetch_seen.add(canonical)
            fetch_ready.append(canonical)

    resolved_csv_path = config.output_dir / "resolved.csv"
    fetch_ready_csv_path = config.output_dir / "fetch_ready_dois.csv"
    candidates_jsonl_path = config.output_dir / "candidates.jsonl"
    crossref_raw_jsonl_path = config.output_dir / "crossref_raw.jsonl"
    summary_json_path = config.output_dir / "summary.json"

    _write_resolved_csv(resolved_csv_path, resolved_rows)
    _write_fetch_ready_csv(fetch_ready_csv_path, fetch_ready)
    _write_jsonl(candidates_jsonl_path, candidates_rows)
    _write_jsonl(crossref_raw_jsonl_path, state.raw_logs)

    summary = {
        "total_rows": len(input_rows),
        "resolved_rows": len(resolved_rows),
        "fetch_ready_count": len(fetch_ready),
        "status_counts": status_counts,
        "method_counts": method_counts,
        "agency_counts": agency_counts,
        "paths": {
            "resolved_csv": str(resolved_csv_path),
            "fetch_ready_csv": str(fetch_ready_csv_path),
            "candidates_jsonl": str(candidates_jsonl_path),
            "crossref_raw_jsonl": str(crossref_raw_jsonl_path),
            "summary_json": str(summary_json_path),
            "cache_json": str(cache_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    _write_cache(cache_path, state.cache, config)
    return summary["paths"]


def _resolve_row(row: dict[str, str], config: DoiResolutionConfig, state: _ResolverState) -> dict[str, Any]:
    candidates = extract_doi_candidates(row["doi_raw"], row["url_raw"], row["title"])
    candidates_payload: list[dict[str, Any]] = []
    best_row = dict(row)
    best_row.update(
        {
            "doi_extracted": candidates[0] if candidates else "",
            "doi_canonical": "",
            "doi_agency": "",
            "doi_status": "",
            "doi_match_method": "manual_none",
            "doi_confidence": "0.0",
            "doi_match_notes": "",
            "crossref_candidate_dois": "[]",
        }
    )

    for candidate in candidates:
        out = _canonicalize_candidate(
            candidate=candidate,
            row=best_row,
            method="extracted",
            inferred=False,
            config=config,
            state=state,
        )
        if out is not None:
            return {"row": out, "candidates": candidates_payload}

    if row["url_raw"] and (not row["title"] or not row["first_author"] or not row["year"]):
        scraped = _fetch_landing_metadata(row["url_raw"], config, state)
        for key in ("title", "first_author", "year", "container_title"):
            if not best_row.get(key) and scraped.get(key):
                best_row[key] = scraped[key]

    has_queryable_fields = bool(best_row["title"] or best_row["first_author"] or best_row["year"] or best_row["container_title"])
    if has_queryable_fields:
        query_items = _crossref_search(best_row, config, state)
        scored = _score_candidates(best_row, query_items)
        candidates_payload.extend(scored)
        best_row["crossref_candidate_dois"] = json.dumps(
            [
                {
                    "doi": cand["doi"],
                    "score": cand["score"],
                    "title_similarity": cand["title_similarity"],
                    "author_match": cand["author_match"],
                    "year_match": cand["year_match"],
                    "type_preference": cand["type_preference"],
                    "container_similarity": cand["container_similarity"],
                }
                for cand in scored
            ],
            ensure_ascii=True,
        )
        if scored:
            top = scored[0]
            second = scored[1] if len(scored) > 1 else None
            gap = top["score"] - (second["score"] if second is not None else 0.0)
            auto_accept = (
                top["score"] >= 0.88
                and top["title_similarity"] >= 0.92
                and gap >= 0.08
                and (not best_row["first_author"] or top["author_match"] > 0.0)
            )
            confirmed_triplet = _is_confirmed_bibliographic_match(best_row, top)
            balanced_match = _is_balanced_bibliographic_match(best_row, top, gap)
            if auto_accept or confirmed_triplet or balanced_match:
                out = _canonicalize_candidate(
                    candidate=top["doi"],
                    row=best_row,
                    method="crossref_search",
                    inferred=True,
                    inferred_confidence=top["score"],
                    config=config,
                    state=state,
                )
                if out is not None:
                    return {"row": out, "candidates": candidates_payload}
            best_row["doi_status"] = "needs_review"
            best_row["doi_match_method"] = "crossref_search"
            best_row["doi_confidence"] = _fmt_float(top["score"])
            best_row["doi_match_notes"] = "ambiguous: top candidates too close or below threshold"
            return {"row": best_row, "candidates": candidates_payload}

    if candidates:
        best_row["doi_status"] = "not_found"
        best_row["doi_match_method"] = "extracted"
        best_row["doi_match_notes"] = "doi candidate could not be canonicalized"
        return {"row": best_row, "candidates": candidates_payload}

    if has_queryable_fields:
        best_row["doi_status"] = "not_found"
        best_row["doi_match_method"] = "crossref_search"
        best_row["doi_match_notes"] = "no matching Crossref candidates"
        return {"row": best_row, "candidates": candidates_payload}

    best_row["doi_status"] = "invalid_input"
    best_row["doi_match_method"] = "manual_none"
    best_row["doi_match_notes"] = "missing DOI, URL, and bibliographic fields"
    return {"row": best_row, "candidates": candidates_payload}


def _canonicalize_candidate(
    *,
    candidate: str,
    row: dict[str, str],
    method: str,
    inferred: bool,
    config: DoiResolutionConfig,
    state: _ResolverState,
    inferred_confidence: float = 1.0,
) -> dict[str, str] | None:
    agency = _crossref_agency(candidate, config, state)
    agency_id = agency.get("agency", "")
    doi_agency = agency_id or "unknown"
    candidate_normalized = normalize_doi(agency.get("doi", "") or candidate)
    if not candidate_normalized:
        return None

    if agency_id == "crossref":
        exists = _crossref_head_exists(candidate_normalized, config, state)
        if not exists:
            return None
        work = _crossref_work(candidate_normalized, config, state)
        canonical = normalize_doi(str(work.get("DOI", "") or candidate_normalized))
        out = dict(row)
        out["doi_canonical"] = canonical
        out["doi_agency"] = doi_agency
        out["doi_status"] = "inferred_crossref" if inferred else "canonicalized_crossref"
        out["doi_match_method"] = method
        out["doi_confidence"] = _fmt_float(inferred_confidence if inferred else 1.0)
        out["doi_match_notes"] = ""
        _fill_row_from_work(out, work)
        return out

    if agency_id:
        resolver = _doi_org_lookup(candidate_normalized, config, state)
        canonical = normalize_doi(str(resolver.get("DOI", "") or candidate_normalized))
        out = dict(row)
        out["doi_canonical"] = canonical
        out["doi_agency"] = doi_agency
        out["doi_status"] = "canonicalized_non_crossref"
        out["doi_match_method"] = method
        out["doi_confidence"] = _fmt_float(0.95)
        out["doi_match_notes"] = ""
        _fill_row_from_work(out, resolver)
        return out
    return None


def _fill_row_from_work(row: dict[str, str], work: dict[str, Any]) -> None:
    if not row.get("title"):
        title_values = _as_list(work.get("title"))
        if title_values:
            row["title"] = str(title_values[0]).strip()
    if not row.get("first_author"):
        authors = _as_list(work.get("author"))
        first = _author_family(authors[0] if authors else {})
        if first:
            row["first_author"] = first
    if not row.get("year"):
        row["year"] = _extract_work_year(work)
    if not row.get("container_title"):
        container = _as_list(work.get("container-title"))
        if container:
            row["container_title"] = str(container[0]).strip()


def _crossref_agency(doi: str, config: DoiResolutionConfig, state: _ResolverState) -> dict[str, str]:
    path = f"/works/{quote(doi, safe='')}/agency"
    url = _crossref_url(path, {}, config.crossref_mailto)
    result = _http_request(
        method="GET",
        url=url,
        config=config,
        state=state,
        request_headers={"Accept": "application/json"},
    )
    if result.status != 200:
        return {}
    payload = _parse_json(result.body)
    message = payload.get("message", {}) if isinstance(payload, dict) else {}
    if not isinstance(message, dict):
        return {}
    agency = message.get("agency", {})
    if not isinstance(agency, dict):
        agency = {}
    return {
        "doi": str(message.get("DOI", "")).strip(),
        "agency": str(agency.get("id", "")).strip().lower(),
    }


def _crossref_head_exists(doi: str, config: DoiResolutionConfig, state: _ResolverState) -> bool:
    path = f"/works/{quote(doi, safe='')}"
    url = _crossref_url(path, {}, config.crossref_mailto)
    result = _http_request(method="HEAD", url=url, config=config, state=state)
    return result.status == 200


def _crossref_work(doi: str, config: DoiResolutionConfig, state: _ResolverState) -> dict[str, Any]:
    path = f"/works/{quote(doi, safe='')}"
    # Crossref singleton work route does not support select parameter.
    url = _crossref_url(path, {}, config.crossref_mailto)
    result = _http_request(
        method="GET",
        url=url,
        config=config,
        state=state,
        request_headers={"Accept": "application/json"},
    )
    if result.status != 200:
        return {}
    payload = _parse_json(result.body)
    message = payload.get("message", {}) if isinstance(payload, dict) else {}
    return message if isinstance(message, dict) else {}


def _crossref_search(row: dict[str, str], config: DoiResolutionConfig, state: _ResolverState) -> list[dict[str, Any]]:
    pieces = [_searchable_title(row.get("title", "")), row.get("year", ""), row.get("container_title", "")]
    query_bibliographic = " ".join(piece.strip() for piece in pieces if piece.strip())
    if not query_bibliographic:
        return []
    author = row.get("first_author", "")
    include_author_options = [True, False] if author else [False]
    for include_author in include_author_options:
        for filter_value in _crossref_filter_variants(row.get("year", "")):
            params: dict[str, str] = {
                "query.bibliographic": query_bibliographic,
                "rows": str(max(int(config.rows), 1)),
                "sort": "score",
                "order": "desc",
                "select": SEARCH_SELECT_FIELDS,
            }
            if include_author and author:
                params["query.author"] = author
            if filter_value:
                params["filter"] = filter_value
            path = "/works"
            url = _crossref_url(path, params, config.crossref_mailto)
            result = _http_request(
                method="GET",
                url=url,
                config=config,
                state=state,
                request_headers={"Accept": "application/json"},
            )
            if result.status != 200:
                continue
            payload = _parse_json(result.body)
            message = payload.get("message", {}) if isinstance(payload, dict) else {}
            if not isinstance(message, dict):
                continue
            items = [item for item in _as_list(message.get("items", [])) if isinstance(item, dict)]
            if items:
                return items
    return []


def _doi_org_lookup(doi: str, config: DoiResolutionConfig, state: _ResolverState) -> dict[str, Any]:
    url = f"https://doi.org/{quote(doi, safe='')}"
    result = _http_request(
        method="GET",
        url=url,
        config=config,
        state=state,
        request_headers={
            "Accept": "application/vnd.citationstyles.csl+json, application/json;q=0.9, text/html;q=0.8",
        },
    )
    if result.status != 200:
        return {}
    parsed = _parse_json(result.body)
    if isinstance(parsed, dict):
        if "DOI" not in parsed:
            parsed["DOI"] = doi
        return parsed
    meta = parse_landing_metadata(result.body)
    return {
        "DOI": doi,
        "title": [meta.get("title", "")] if meta.get("title") else [],
        "author": [{"family": meta.get("first_author", "")}] if meta.get("first_author") else [],
        "issued": {"date-parts": [[int(meta["year"])]]} if meta.get("year") else {},
        "container-title": [meta.get("container_title", "")] if meta.get("container_title") else [],
    }


def _fetch_landing_metadata(url: str, config: DoiResolutionConfig, state: _ResolverState) -> dict[str, str]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {"title": "", "first_author": "", "year": "", "container_title": ""}
    result = _http_request(
        method="GET",
        url=url,
        config=config,
        state=state,
        request_headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
    )
    if result.status != 200:
        return {"title": "", "first_author": "", "year": "", "container_title": ""}
    return parse_landing_metadata(result.body)


def _http_request(
    *,
    method: str,
    url: str,
    config: DoiResolutionConfig,
    state: _ResolverState,
    request_headers: dict[str, str] | None = None,
) -> _HttpResult:
    cache_key = f"{method} {url}"
    if config.use_cache and not config.refresh_cache and cache_key in state.cache:
        cached = state.cache[cache_key]
        result = _HttpResult(
            method=method,
            url=url,
            status=int(cached.get("status", 0)),
            body=str(cached.get("body", "")),
            headers={str(k): str(v) for k, v in dict(cached.get("headers", {})).items()},
            from_cache=True,
            error=str(cached.get("error", "")),
        )
        _log_http(result, state)
        return result

    headers = {"User-Agent": _user_agent(config), **(request_headers or {})}
    retries = max(int(config.max_retries), 0)
    wait_seconds = 0.5
    last_result = _HttpResult(method=method, url=url, status=0, body="", headers={})
    for attempt in range(retries + 1):
        req = request.Request(url=url, headers=headers, method=method)
        try:
            with config.urlopen(req, timeout=config.timeout) as resp:
                status = int(getattr(resp, "status", 200))
                body = ""
                if method != "HEAD":
                    body = resp.read().decode("utf-8", errors="replace")
                last_result = _HttpResult(
                    method=method,
                    url=url,
                    status=status,
                    body=body,
                    headers=_headers_dict(getattr(resp, "headers", {})),
                )
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_result = _HttpResult(
                method=method,
                url=url,
                status=int(getattr(exc, "code", 0) or 0),
                body=body,
                headers=_headers_dict(getattr(exc, "headers", {})),
                error=str(exc),
            )
        except URLError as exc:
            last_result = _HttpResult(
                method=method,
                url=url,
                status=0,
                body="",
                headers={},
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            last_result = _HttpResult(
                method=method,
                url=url,
                status=0,
                body="",
                headers={},
                error=str(exc),
            )

        retryable = last_result.status == 429 or last_result.status >= 500 or last_result.status == 0
        if retryable and attempt < retries:
            time.sleep(wait_seconds)
            wait_seconds *= 2.0
            continue
        break

    _log_http(last_result, state)
    if config.use_cache:
        state.cache[cache_key] = {
            "status": last_result.status,
            "body": last_result.body,
            "headers": last_result.headers,
            "error": last_result.error,
        }
    return last_result


def _log_http(result: _HttpResult, state: _ResolverState) -> None:
    log = {
        "method": result.method,
        "url": result.url,
        "status": result.status,
        "from_cache": result.from_cache,
        "error": result.error,
        "response": result.body,
    }
    state.raw_logs.append(log)


def _score_candidates(row: dict[str, str], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for item in items:
        doi = normalize_doi(str(item.get("DOI", "")))
        if not doi:
            continue
        item_title = str((_as_list(item.get("title")) or [""])[0])
        title_similarity = _title_similarity(row.get("title", ""), item_title)
        author_match = _author_similarity(row.get("first_author", ""), item.get("author"))
        year_match = _year_similarity(_year_to_int(row.get("year", "")), _extract_work_year(item))
        type_preference = 1.0 if str(item.get("type", "")).strip().lower() == "journal-article" else 0.0
        candidate_container = str((_as_list(item.get("container-title")) or [""])[0])
        container_similarity = _container_similarity(row.get("container_title", ""), candidate_container)
        score = (
            0.55 * title_similarity
            + 0.20 * author_match
            + 0.15 * year_match
            + 0.10 * type_preference
        )
        scored.append(
            {
                "doi": doi,
                "score": round(score, 6),
                "title_similarity": round(title_similarity, 6),
                "author_match": round(author_match, 6),
                "year_match": round(year_match, 6),
                "type_preference": round(type_preference, 6),
                "container_similarity": round(container_similarity, 6),
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), item["doi"]))
    return scored


def _title_similarity(left: str, right: str) -> float:
    a = _norm_text(left)
    b = _norm_text(right)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    left_tokens = set(a.split())
    right_tokens = set(b.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return max(0.0, min(1.0, (seq + jaccard) / 2.0))


def _container_similarity(left: str, right: str) -> float:
    a = _norm_text(left)
    b = _norm_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _author_similarity(input_author: str, candidate_authors: Any) -> float:
    query = _author_last_name(input_author)
    if not query:
        return 0.0
    for author in _as_list(candidate_authors):
        family = _author_family(author)
        if not family:
            continue
        if family == query:
            return 1.0
        if SequenceMatcher(None, family, query).ratio() >= 0.8:
            return 0.5
    return 0.0


def _year_similarity(input_year: int | None, candidate_year_raw: str) -> float:
    candidate_year = _year_to_int(candidate_year_raw)
    if input_year is None or candidate_year is None:
        return 0.0
    if input_year == candidate_year:
        return 1.0
    if abs(input_year - candidate_year) == 1:
        return 0.5
    return 0.0


def _normalize_row(row: dict[str, Any], row_index: int, columns: dict[str, str]) -> dict[str, str]:
    doi_raw = str(row.get(columns.get("doi", ""), "") or "").strip()
    url_raw = str(row.get(columns.get("url", ""), "") or "").strip()
    title = str(row.get(columns.get("title", ""), "") or "").strip()
    author_raw = str(row.get(columns.get("author", ""), "") or "").strip()
    year_raw = str(row.get(columns.get("year", ""), "") or "").strip()
    container = str(row.get(columns.get("container", ""), "") or "").strip()
    return {
        "row_index": str(row_index),
        "doi_raw": doi_raw,
        "url_raw": url_raw,
        "title": title,
        "first_author": _first_author(author_raw),
        "year": _extract_year(year_raw),
        "container_title": container,
    }


def _resolve_columns(fieldnames: list[str], config: DoiResolutionConfig) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in fieldnames:
        low = name.strip().lower()
        if low and low not in lookup:
            lookup[low] = name

    def _pick(override: str | None, aliases: tuple[str, ...]) -> str:
        if override:
            candidate = lookup.get(override.strip().lower(), "")
            if candidate:
                return candidate
            raise ValueError(f"Requested column '{override}' not found in input CSV.")
        for alias in aliases:
            candidate = lookup.get(alias.lower(), "")
            if candidate:
                return candidate
        return ""

    return {
        "doi": _pick(config.doi_column, INPUT_ALIASES["doi"]),
        "url": _pick(config.url_column, INPUT_ALIASES["url"]),
        "title": _pick(config.title_column, INPUT_ALIASES["title"]),
        "author": _pick(config.author_column, INPUT_ALIASES["author"]),
        "year": _pick(config.year_column, INPUT_ALIASES["year"]),
        "container": _pick(config.container_column, INPUT_ALIASES["container"]),
    }


def _extract_year(raw: str) -> str:
    match = YEAR_RE.search(raw or "")
    return match.group(0) if match else ""


def _year_to_int(raw: str) -> int | None:
    text = _extract_year(raw)
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _first_author(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.split(r"\bet\.?\s*al\.?\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    if _is_unknown_value(text):
        return ""
    return text


def _author_last_name(raw: str) -> str:
    text = _first_author(raw).lower()
    if not text:
        return ""
    if "," in text:
        return text.split(",", 1)[0].strip()
    pieces = [piece.strip() for piece in text.split() if piece.strip()]
    return pieces[-1] if pieces else ""


def _is_unknown_value(raw: str) -> bool:
    norm = (raw or "").strip().lower()
    if not norm:
        return True
    return norm in UNKNOWN_MARKERS


def _searchable_title(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    # Strip synthetic prefixes and source DOI annotations from derived corpus rows.
    text = re.sub(r"^hs-\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(source\s+10\.[^)]+\)", "", text, flags=re.IGNORECASE)
    text = text.replace("...", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _crossref_filter_variants(year_raw: str) -> list[str]:
    year = _year_to_int(year_raw)
    if year is None:
        return ["type:journal-article", ""]
    return [
        f"from-pub-date:{year},until-pub-date:{year},type:journal-article",
        f"from-pub-date:{year - 1},until-pub-date:{year + 1},type:journal-article",
        f"from-pub-date:{year},until-pub-date:{year}",
        "type:journal-article",
        "",
    ]


def _is_confirmed_bibliographic_match(row: dict[str, str], candidate: dict[str, Any]) -> bool:
    has_required = bool(row.get("title") and row.get("first_author") and row.get("year"))
    if not has_required:
        return False
    if float(candidate.get("title_similarity", 0.0)) < 0.92:
        return False
    if float(candidate.get("author_match", 0.0)) < 1.0:
        return False
    if float(candidate.get("year_match", 0.0)) < 0.5:
        return False
    container = (row.get("container_title", "") or "").strip()
    if container:
        if float(candidate.get("container_similarity", 0.0)) < 0.6:
            return False
    return True


def _is_balanced_bibliographic_match(row: dict[str, str], candidate: dict[str, Any], gap: float) -> bool:
    score = float(candidate.get("score", 0.0))
    title_similarity = float(candidate.get("title_similarity", 0.0))
    author_match = float(candidate.get("author_match", 0.0))
    year_match = float(candidate.get("year_match", 0.0))
    container_similarity = float(candidate.get("container_similarity", 0.0))
    has_author = bool((row.get("first_author", "") or "").strip())

    # Track A: author present, rely on author/year confirmation with moderate title threshold.
    if has_author:
        return (
            score >= 0.62
            and title_similarity >= 0.38
            and author_match >= 1.0
            and year_match >= 1.0
            and gap >= 0.03
        )

    # Track B: author missing/unknown, require stronger title+container evidence.
    return (
        score >= 0.50
        and title_similarity >= 0.46
        and year_match >= 1.0
        and container_similarity >= 0.30
        and gap >= 0.04
    )


def _author_family(author: Any) -> str:
    if not isinstance(author, dict):
        return ""
    family = str(author.get("family", "") or "").strip().lower()
    if family:
        return family
    name = str(author.get("name", "") or "").strip().lower()
    if not name:
        return ""
    pieces = name.split()
    return pieces[-1] if pieces else ""


def _extract_work_year(work: dict[str, Any]) -> str:
    for key in ("issued", "published-print", "published-online", "created"):
        value = work.get(key)
        if isinstance(value, dict):
            date_parts = value.get("date-parts")
            if isinstance(date_parts, list) and date_parts:
                first = date_parts[0]
                if isinstance(first, list) and first:
                    try:
                        year = int(first[0])
                        if 1900 <= year <= 2100:
                            return str(year)
                    except Exception:
                        continue
    return ""


def _jsonld_first_author(authors: Any) -> str:
    first = _as_list(authors)
    if not first:
        return ""
    item = first[0]
    if isinstance(item, dict):
        return str(item.get("name", "")).strip()
    return str(item).strip()


def _norm_text(text: str) -> str:
    lowered = unescape((text or "")).lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _headers_dict(headers: Any) -> dict[str, str]:
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    if hasattr(headers, "items"):
        try:
            return {str(k): str(v) for k, v in headers.items()}
        except Exception:
            return {}
    return {}


def _write_resolved_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESOLVED_COLUMNS)
        writer.writeheader()
        for row in rows:
            payload = {key: str(row.get(key, "")) for key in RESOLVED_COLUMNS}
            writer.writerow(payload)


def _write_fetch_ready_csv(path: Path, dois: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["DOI"])
        writer.writeheader()
        for doi in dois:
            writer.writerow({"DOI": doi})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def _load_cache(path: Path, config: DoiResolutionConfig) -> dict[str, dict[str, Any]]:
    if not config.use_cache or config.refresh_cache or not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in loaded.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _write_cache(path: Path, cache: dict[str, dict[str, Any]], config: DoiResolutionConfig) -> None:
    if not config.use_cache:
        return
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True))


def _parse_json(raw: str) -> dict[str, Any] | list[Any] | str:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    return parsed


def _crossref_url(path: str, params: dict[str, str], mailto: str) -> str:
    payload = dict(params)
    if mailto.strip():
        payload["mailto"] = mailto.strip()
    query = urlencode(payload, doseq=True)
    if query:
        return f"https://api.crossref.org{path}?{query}"
    return f"https://api.crossref.org{path}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fmt_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _user_agent(config: DoiResolutionConfig) -> str:
    base = config.user_agent.strip() or "paper-ocr/0.1"
    mailto = config.crossref_mailto.strip()
    if not mailto or mailto in base:
        return base
    return f"{base} (mailto:{mailto})"
