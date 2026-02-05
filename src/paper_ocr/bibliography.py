from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


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


def bibliography_prompt(first_page_markdown: str) -> str:
    return (
        "You are extracting citation metadata from the first page of an academic paper.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        '{'
        '"title": "string", '
        '"authors": ["string"], '
        '"year": "YYYY", '
        '"journal_ref": "string", '
        '"doi": "string"'
        "}\n"
        "Rules:\n"
        "- Use authors in display order.\n"
        "- Keep `journal_ref` concise, e.g. journal, volume(issue), pages.\n"
        "- `doi` should be bare DOI without URL prefix when available.\n"
        "- If a field is unknown, return an empty string (or empty list for authors).\n\n"
        "First page markdown:\n"
        f"{first_page_markdown}"
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
