from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def evaluate_table_pipeline(gold_dir: Path, pred_dir: Path) -> dict[str, Any]:
    gold_rows = _load_jsonl(gold_dir / "tables.jsonl")
    pred_rows = _load_jsonl(pred_dir / "tables.jsonl")

    gold_pages = {int(r.get("page", 0)) for r in gold_rows if int(r.get("page", 0)) > 0}
    pred_pages = {int(r.get("page", 0)) for r in pred_rows if int(r.get("page", 0)) > 0}
    tp = len(gold_pages & pred_pages)
    fp = len(pred_pages - gold_pages)
    fn = len(gold_pages - pred_pages)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    def _headers(row: dict[str, Any]) -> list[str]:
        values = row.get("headers", [])
        if not isinstance(values, list):
            return []
        return [str(v) for v in values]

    def _rows(row: dict[str, Any]) -> list[list[str]]:
        values = row.get("rows", [])
        if not isinstance(values, list):
            return []
        out: list[list[str]] = []
        for item in values:
            if isinstance(item, list):
                out.append([str(v) for v in item])
        return out

    def _norm_text(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def _extract_float(value: str) -> float | None:
        text = str(value or "").replace(",", "").strip()
        if not text:
            return None
        match = num_re.search(text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    gold_by_page: dict[int, list[dict[str, Any]]] = {}
    pred_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in gold_rows:
        page = int(row.get("page", 0) or 0)
        if page > 0:
            gold_by_page.setdefault(page, []).append(row)
    for row in pred_rows:
        page = int(row.get("page", 0) or 0)
        if page > 0:
            pred_by_page.setdefault(page, []).append(row)
    for rows in gold_by_page.values():
        rows.sort(key=lambda r: str(r.get("table_id", "")))
    for rows in pred_by_page.values():
        rows.sort(key=lambda r: str(r.get("table_id", "")))

    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for page in sorted(set(gold_by_page).intersection(pred_by_page)):
        g = gold_by_page.get(page, [])
        p = pred_by_page.get(page, [])
        for idx in range(min(len(g), len(p))):
            matched_pairs.append((g[idx], p[idx]))

    row_count_matches = 0
    col_count_matches = 0
    key_cell_total = 0
    key_cell_match = 0
    numeric_total = 0
    numeric_match = 0
    for gold_row, pred_row in matched_pairs:
        g_headers = _headers(gold_row)
        p_headers = _headers(pred_row)
        g_rows = _rows(gold_row)
        p_rows = _rows(pred_row)
        if len(g_rows) == len(p_rows):
            row_count_matches += 1
        if len(g_headers) == len(p_headers):
            col_count_matches += 1

        max_rows = min(len(g_rows), len(p_rows))
        max_cols = min(len(g_headers), len(p_headers))
        for r_idx in range(max_rows):
            for c_idx in range(max_cols):
                g_cell = _norm_text(g_rows[r_idx][c_idx] if c_idx < len(g_rows[r_idx]) else "")
                p_cell = _norm_text(p_rows[r_idx][c_idx] if c_idx < len(p_rows[r_idx]) else "")
                key_cell_total += 1
                if g_cell == p_cell:
                    key_cell_match += 1
                g_num = _extract_float(g_cell)
                if g_num is None:
                    continue
                numeric_total += 1
                p_num = _extract_float(p_cell)
                if p_num is not None:
                    numeric_match += 1

    pair_count = len(matched_pairs)
    row_count_match_rate = row_count_matches / max(pair_count, 1)
    column_count_match_rate = col_count_matches / max(pair_count, 1)
    key_cell_accuracy = key_cell_match / max(key_cell_total, 1)
    numeric_parse_success = numeric_match / max(numeric_total, 1)

    return {
        "gold_table_count": len(gold_rows),
        "pred_table_count": len(pred_rows),
        "matched_page_count": tp,
        "table_detection_precision": precision,
        "table_detection_recall": recall,
        "matched_table_count": pair_count,
        "row_count_match_rate": row_count_match_rate,
        "column_count_match_rate": column_count_match_rate,
        "key_cell_accuracy": key_cell_accuracy,
        "numeric_parse_success": numeric_parse_success,
    }
