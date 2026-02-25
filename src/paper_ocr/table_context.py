from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{0,11}$")
UNITISH_RE = re.compile(r"^(?:pa|wt|mol|kg|g|mg|ml|l|m|mm|cm|nm|s|ms|hz|k|c)$", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{0,11}")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _is_code_token(token: str) -> bool:
    value = _normalize_text(token)
    if not value:
        return False
    if not CODE_RE.match(value):
        return False
    if value.isdigit():
        return False
    if UNITISH_RE.match(value):
        return False
    letters = sum(1 for ch in value if ch.isalpha())
    digits = sum(1 for ch in value if ch.isdigit())
    upper_letters = sum(1 for ch in value if ch.isalpha() and ch.isupper())
    if digits == 0 and letters <= 1:
        return False
    if letters > 0 and (upper_letters / float(letters)) < 0.5:
        return False
    return True


def _extract_candidate_codes(table: dict[str, Any]) -> list[str]:
    headers = [str(x) for x in table.get("headers", [])] if isinstance(table.get("headers"), list) else []
    rows = [row for row in table.get("rows", []) if isinstance(row, list)]

    candidates: set[str] = set()
    for row in rows:
        if not row:
            continue
        token = _normalize_text(str(row[0]))
        if _is_code_token(token):
            candidates.add(token)

    header_hint = " ".join(headers).lower()
    if any(h in header_hint for h in ("code", "sample", "id", "polymer")):
        for row in rows:
            if not row:
                continue
            for token in TOKEN_RE.findall(_normalize_text(str(row[0]))):
                if _is_code_token(token):
                    candidates.add(token)

    return sorted(candidates)


def _nearby_lines(doc_dir: Path, page: int, window: int = 1) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    start = max(1, int(page) - int(window))
    end = int(page) + int(window)
    for p in range(start, end + 1):
        page_path = doc_dir / "pages" / f"{p:04d}.md"
        if not page_path.exists():
            continue
        for line_idx, raw in enumerate(page_path.read_text().splitlines(), start=1):
            text = _normalize_text(raw)
            if text:
                lines.append((p, line_idx, text))
    return lines


def _extract_definitions(lines: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page, line_idx, line in lines:
        m_eq = re.match(r"^([A-Za-z][A-Za-z0-9\-]{0,11})\s*[:=]\s*(.+)$", line)
        if m_eq and _is_code_token(m_eq.group(1)):
            out.append(
                {
                    "code": m_eq.group(1),
                    "resolved_text": _normalize_text(m_eq.group(2)).rstrip(".,;"),
                    "confidence": 0.95,
                    "source_page": page,
                    "source_line": line_idx,
                    "match_type": "equals",
                }
            )

        for code, desc in re.findall(r"\b([A-Za-z][A-Za-z0-9\-]{0,11})\s*[:=]\s*([^;]+)", line):
            if not _is_code_token(code):
                continue
            out.append(
                {
                    "code": code,
                    "resolved_text": _normalize_text(desc).rstrip(".,;"),
                    "confidence": 0.88,
                    "source_page": page,
                    "source_line": line_idx,
                    "match_type": "inline_equals",
                }
            )

        for desc, code in re.findall(r"([A-Za-z][A-Za-z0-9\s,\-]{2,80})\(\s*([A-Za-z][A-Za-z0-9\-]{0,11})\s*\)", line):
            if not _is_code_token(code):
                continue
            out.append(
                {
                    "code": code,
                    "resolved_text": _normalize_text(desc).rstrip(".,;"),
                    "confidence": 0.75,
                    "source_page": page,
                    "source_line": line_idx,
                    "match_type": "paren_reverse",
                }
            )
    return out


def resolve_table_context_mappings(*, doc_dir: Path, page: int, table: dict[str, Any], window: int = 1) -> dict[str, Any]:
    codes = _extract_candidate_codes(table)
    if not codes:
        return {"candidate_codes": [], "mappings": [], "unresolved_codes": [], "evidence_pages": []}

    lines = _nearby_lines(doc_dir=doc_dir, page=page, window=window)
    definitions = _extract_definitions(lines)
    by_code: dict[str, dict[str, Any]] = {}
    for row in definitions:
        code = str(row.get("code", ""))
        if code not in codes:
            continue
        resolved = _normalize_text(str(row.get("resolved_text", "")))
        if not resolved:
            continue
        existing = by_code.get(code)
        if existing is None or float(row.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            by_code[code] = row

    mappings = [by_code[code] for code in sorted(by_code)]
    unresolved = [code for code in codes if code not in by_code]
    evidence_pages = sorted({int(row.get("source_page", 0) or 0) for row in mappings if int(row.get("source_page", 0) or 0) > 0})
    return {
        "candidate_codes": codes,
        "mappings": mappings,
        "unresolved_codes": unresolved,
        "evidence_pages": evidence_pages,
    }

