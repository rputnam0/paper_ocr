from __future__ import annotations

import json
from pathlib import Path
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

    return {
        "gold_table_count": len(gold_rows),
        "pred_table_count": len(pred_rows),
        "matched_page_count": tp,
        "table_detection_precision": precision,
        "table_detection_recall": recall,
    }

