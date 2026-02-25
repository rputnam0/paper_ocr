from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
AFFILIATION_HINTS = (
    "university",
    "department",
    "school",
    "laboratory",
    "institute",
    "college",
    "center",
    "centre",
    "hospital",
)


@dataclass
class BibliographyInfo:
    title: str
    authors: list[str]
    year: str
    journal_ref: str
    doi: str


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def normalize_bibliography(raw: dict[str, Any]) -> BibliographyInfo:
    title = str(raw.get("title", "")).strip()
    authors_raw = raw.get("authors", [])
    if isinstance(authors_raw, str):
        authors = [authors_raw.strip()] if authors_raw.strip() else []
    elif isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = []

    year_value = str(raw.get("year", "")).strip()
    year_match = re.search(r"\b(19|20)\d{2}\b", year_value)
    year = year_match.group(0) if year_match else ""

    journal_ref = str(raw.get("journal_ref", "")).strip()
    doi = str(raw.get("doi", "")).strip()
    return BibliographyInfo(
        title=title,
        authors=authors,
        year=year,
        journal_ref=journal_ref,
        doi=doi,
    )


def _snake(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _author_token(author: str) -> str:
    if "," in author:
        last, given = [p.strip() for p in author.split(",", 1)]
        given_tokens = re.findall(r"[A-Za-z]+", given)
        pieces = [last] + given_tokens
    else:
        parts = [p for p in re.split(r"\s+", author.strip()) if p]
        if not parts:
            pieces = []
        elif len(parts) == 1:
            pieces = parts
        else:
            pieces = [parts[-1], *parts[:-1]]
    return _snake("_".join(pieces))


def folder_name_from_bibliography(info: BibliographyInfo) -> str:
    author_part = _author_token(info.authors[0]) if info.authors else "unknown_author"
    year_part = info.year or "unknown_year"
    out = _snake(f"{author_part}_{year_part}")
    return out or "unknown_paper"


def markdown_filename_from_title(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', " ", title).strip().rstrip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return f"{cleaned or 'document'}.md"


def citation_from_bibliography(info: BibliographyInfo) -> str:
    parts: list[str] = []
    if info.journal_ref:
        parts.append(info.journal_ref)
    if info.doi:
        parts.append(f"doi:{info.doi}")
    return " ".join(parts).strip()


def _clean_markdown_line(raw: str) -> str:
    line = str(raw or "").strip()
    if not line:
        return ""
    line = re.sub(r"^#{1,6}\s+", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"[`*_~]+", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _looks_like_person_name(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 3 or len(text) > 70:
        return False
    if "@" in text or "http" in text.lower() or any(ch.isdigit() for ch in text):
        return False
    low = text.lower()
    if any(hint in low for hint in AFFILIATION_HINTS):
        return False
    tokens = [tok.strip(".,*") for tok in re.split(r"\s+", text) if tok.strip(".,*")]
    if len(tokens) < 2 or len(tokens) > 5:
        return False
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            continue
        if not re.match(r"^[A-Z][A-Za-z'`-]*$", tok):
            return False
    return True


def _authors_from_line(line: str) -> list[str]:
    text = str(line or "").strip()
    if not text:
        return []
    text = re.sub(r"\b(and|&)\b", ",", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    if len(parts) == 1 and _looks_like_person_name(parts[0]):
        return [parts[0]]
    out: list[str] = []
    for part in parts:
        candidate = part.strip(" *")
        if _looks_like_person_name(candidate):
            out.append(candidate)
    # Preserve display order while deduplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for author in out:
        key = author.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(author)
    return deduped


def _is_probable_title(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith(("abstract", "keywords", "introduction", "doi")):
        return False
    if "@" in text or "http" in low or "doi:" in low:
        return False
    words = [w for w in re.split(r"\s+", text) if w]
    if len(words) < 3 or len(text) < 16:
        return False
    if len(text) > 220:
        return False
    return True


def _extract_title(lines: list[str]) -> tuple[str, int]:
    for idx, line in enumerate(lines[:30]):
        if _is_probable_title(line):
            return line, idx
    return "", 0


def _extract_authors(lines: list[str], title_idx: int) -> list[str]:
    if not lines:
        return []
    stop_words = ("abstract", "keywords", "introduction")
    start = min(max(title_idx + 1, 0), len(lines))
    end = min(start + 12, len(lines))
    for line in lines[start:end]:
        low = line.lower()
        if low.startswith(stop_words):
            break
        authors = _authors_from_line(line)
        if len(authors) >= 2:
            return authors
        if len(authors) == 1 and ("," in line or ";" in line or " and " in low):
            return authors
    return []


def _extract_doi(text: str) -> str:
    match = DOI_RE.search(str(text or ""))
    if not match:
        return ""
    return match.group(0).rstrip(".,;)")


def _extract_year(lines: list[str]) -> str:
    for line in lines:
        if "doi" in line.lower():
            continue
        match = YEAR_RE.search(line)
        if match:
            return match.group(0)
    return ""


def _extract_journal_ref(lines: list[str], title: str, authors: list[str]) -> str:
    title_low = title.lower().strip()
    author_lows = {a.lower().strip() for a in authors}
    hints = ("journal", "proceedings", "conference", "vol.", "volume", "issue", "arxiv", "pp.", "pages")
    for line in lines:
        low = line.lower().strip()
        if not low:
            continue
        if low == title_low or low in author_lows:
            continue
        if "doi" in low:
            continue
        if any(h in low for h in hints) or YEAR_RE.search(line):
            return line
    return ""


def extract_bibliography_deterministic(first_page_markdown: str) -> BibliographyInfo:
    lines = [_clean_markdown_line(line) for line in str(first_page_markdown or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return normalize_bibliography({})
    title, title_idx = _extract_title(lines)
    authors = _extract_authors(lines, title_idx=title_idx)
    year = _extract_year(lines)
    doi = _extract_doi(first_page_markdown)
    journal_ref = _extract_journal_ref(lines, title=title, authors=authors)
    return normalize_bibliography(
        {
            "title": title,
            "authors": authors,
            "year": year,
            "journal_ref": journal_ref,
            "doi": doi,
        }
    )
