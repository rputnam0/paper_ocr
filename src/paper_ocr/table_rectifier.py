from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from .client import call_text_model

SYMBOL_CHAR_RE = re.compile(r"[^\x00-\x7F]|[±≤≥≈×÷°µμδΔητσγβαΩω]")
UNIT_TOKEN_RE = re.compile(r"(?:\bpa\b|\bwt\b|mol|g|kg|mg|ml|dl|l|%|°c|°k|ppm|\bm\b|\bs\b)", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMERICISH_RE = re.compile(r"^[0-9.,+\-−–—/%()±×*]+$")
PROVENANCE_SOURCES = {"marker", "ocr", "context"}
IGNORABLE_SYMBOLS = {"-", "*"}
GROUP_PAGE_TABLE_RE = re.compile(r"page_(\d+)_table_(\d+)$")
CONTEXT_WINDOW_OPTIONS = ("none", "table_page", "page_plus_minus_1", "page_plus_minus_2")


@dataclass
class RectifierConfig:
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 2000
    risk_threshold: float = 0.45
    max_tables_per_doc: int = 12
    target: str = "risk"
    structure_model: str = "off"
    structure_lock: bool = True
    context_mode: str = "on_demand"
    skip_already_rectified: bool = False


@dataclass
class RectifierResult:
    enabled: bool
    model: str
    target: str
    considered: int = 0
    selected: int = 0
    applied: int = 0
    skipped_low_risk: int = 0
    fallbacked: int = 0
    invalid_schema: int = 0
    evidence_violations: int = 0
    errors: int = 0
    report_path: str = ""
    run_id: str = ""
    table_results: list[dict[str, Any]] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _normalize_cell_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _collapse_header_rows(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    max_cols = max((len(row) for row in header_rows), default=0)
    if max_cols <= 0:
        return []
    normalized = [list(row) + [""] * (max_cols - len(row)) for row in header_rows]
    out: list[str] = []
    for col in range(max_cols):
        tokens: list[str] = []
        for row in normalized:
            token = _normalize_cell_text(row[col])
            if not token:
                continue
            if not tokens or tokens[-1] != token:
                tokens.append(token)
        if not tokens:
            out.append("")
        elif len(tokens) == 1:
            out.append(tokens[0])
        else:
            out.append(f"{tokens[0]} ({' / '.join(tokens[1:])})")
    return out


def _extract_symbols(text: str) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for ch in SYMBOL_CHAR_RE.findall(str(text or "")):
        if ch in seen:
            continue
        seen.add(ch)
        ordered.append(ch)
    return "".join(ordered)


def _quality_metrics(headers: list[str], rows: list[list[str]]) -> dict[str, Any]:
    total_cells = max(len(headers), 1) * max(len(rows), 1)
    flattened = [cell for row in rows for cell in row]
    empty_cells = sum(1 for cell in flattened if not str(cell).strip())
    repeated = 0
    if flattened:
        repeated = max((flattened.count(v) for v in set(flattened)))
    row_lengths = [len(row) for row in rows] if rows else [len(headers)]
    unstable = sum(1 for n in row_lengths if n != len(headers))
    return {
        "empty_cell_ratio": empty_cells / max(total_cells, 1),
        "repeated_text_ratio": repeated / max(len(flattened), 1),
        "column_instability_ratio": unstable / max(len(row_lengths), 1),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _make_flag_id(table_id: str, flag_type: str, details: str) -> str:
    src = f"llm_rectifier|{table_id}|{flag_type}|{details}"
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _append_flag(
    *,
    qa_rows: list[dict[str, Any]],
    existing_flag_ids: set[str],
    table_id: str,
    page: int,
    flag_type: str,
    severity: str,
    details: str,
) -> None:
    flag_id = _make_flag_id(table_id, flag_type, details)
    if flag_id in existing_flag_ids:
        return
    qa_rows.append(
        {
            "flag_id": flag_id,
            "severity": severity,
            "type": flag_type,
            "page": int(page),
            "table_ref": table_id or "*",
            "details": details,
        }
    )
    existing_flag_ids.add(flag_id)


def _compute_risk_score(table_payload: dict[str, Any], qa_rows: list[dict[str, Any]]) -> float:
    metrics = table_payload.get("quality_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    quality_component = max(
        float(metrics.get("empty_cell_ratio", 0.0) or 0.0),
        float(metrics.get("repeated_text_ratio", 0.0) or 0.0),
        float(metrics.get("column_instability_ratio", 0.0) or 0.0),
    )
    required_missing = table_payload.get("required_fields_missing", [])
    missing_component = min(float(len(required_missing if isinstance(required_missing, list) else [])) / 3.0, 1.0)
    header_rows_full = table_payload.get("header_rows_full", [])
    header_depth = len(header_rows_full if isinstance(header_rows_full, list) else [])
    if header_depth >= 3:
        header_complexity = 1.0
    elif header_depth == 2:
        header_complexity = 0.6
    else:
        header_complexity = 0.0

    table_id = str(table_payload.get("table_id", ""))
    table_flags = [
        row
        for row in qa_rows
        if str(row.get("table_ref", "")) == table_id and str(row.get("severity", "")).lower() in {"warn", "error"}
    ]
    disagreement_component = min(float(len(table_flags)) / 3.0, 1.0)
    risk = (0.45 * quality_component) + (0.20 * missing_component) + (0.15 * header_complexity) + (0.20 * disagreement_component)
    return round(max(0.0, min(1.0, risk)), 4)


def _collect_target_table_ids(validation_path: Path, *, target: str) -> set[str]:
    out: set[str] = set()
    for row in _load_jsonl(validation_path):
        table_id = str(row.get("table_id", "")).strip()
        review = row.get("model_review", {})
        if not table_id or not isinstance(review, dict):
            continue
        recommended_action = str(review.get("recommended_action", "")).strip().lower()
        final_action = str(review.get("final_action", "")).strip().lower()
        if target == "nonaccept":
            if final_action == "reject" or recommended_action in {"review", "reject"}:
                out.add(table_id)
            continue
        if target == "reject":
            if final_action == "reject" or (not final_action and recommended_action == "reject"):
                out.add(table_id)
            continue
    return out


def _collect_validation_feedback_map(validation_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(validation_path):
        table_id = str(row.get("table_id", "")).strip()
        review = row.get("model_review", {})
        if not table_id or not isinstance(review, dict):
            continue
        feedback = {
            "recommended_action": str(review.get("recommended_action", "")).strip().lower(),
            "final_action": str(review.get("final_action", "")).strip().lower(),
            "failure_modes": [str(x).strip() for x in review.get("failure_modes", []) if str(x).strip()]
            if isinstance(review.get("failure_modes", []), list)
            else [],
            "issues": [str(x).strip() for x in review.get("issues", []) if str(x).strip()]
            if isinstance(review.get("issues", []), list)
            else [],
            "llm_extraction_instructions": [
                str(x).strip() for x in review.get("llm_extraction_instructions", []) if str(x).strip()
            ]
            if isinstance(review.get("llm_extraction_instructions", []), list)
            else [],
            "missing_required_information": [
                str(x).strip() for x in review.get("missing_required_information", []) if str(x).strip()
            ]
            if isinstance(review.get("missing_required_information", []), list)
            else [],
            "formatting_issues": [str(x).strip() for x in review.get("formatting_issues", []) if str(x).strip()]
            if isinstance(review.get("formatting_issues", []), list)
            else [],
            "root_cause_hypothesis": str(review.get("root_cause_hypothesis", "") or "").strip(),
        }
        out[table_id] = feedback
    return out


def _baseline_payload_for_rerun(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    llm_meta = payload.get("llm_rectification", {})
    if not isinstance(llm_meta, dict):
        return out
    snapshot = llm_meta.get("original_snapshot", {})
    if not isinstance(snapshot, dict):
        return out
    if not isinstance(snapshot.get("data_rows"), list):
        return out
    if not isinstance(snapshot.get("header_rows_full"), list):
        return out

    out["header_rows_full"] = snapshot.get("header_rows_full", out.get("header_rows_full", []))
    out["data_rows"] = snapshot.get("data_rows", out.get("data_rows", []))
    out["header_hierarchy"] = snapshot.get("header_hierarchy", out.get("header_hierarchy", []))
    out["row_lineage"] = snapshot.get("row_lineage", out.get("row_lineage", []))
    out["context_mappings"] = snapshot.get("context_mappings", out.get("context_mappings", []))
    out["required_fields_missing"] = snapshot.get("required_fields_missing", out.get("required_fields_missing", []))
    out["quality_metrics"] = snapshot.get("quality_metrics", out.get("quality_metrics", {}))
    out.pop("cell_provenance", None)
    out.pop("rectifier_confidence", None)
    out.pop("rectifier_needs_review", None)
    out.pop("llm_rectification", None)
    return out


def _resolve_fragment_index(doc_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    fragments_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "table_fragments.jsonl"
    for row in _load_jsonl(fragments_path):
        fid = str(row.get("fragment_id", "")).strip()
        if fid:
            out[fid] = row
    return out


def _load_table_structure_map(doc_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    manifest_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "table_structure" / "manifest.jsonl"
    for row in _load_jsonl(manifest_path):
        table_id = str(row.get("table_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        if not table_id or status != "ok":
            continue
        rel_path = str(row.get("structure_path", "")).strip()
        if not rel_path:
            continue
        payload = _load_json(doc_dir / rel_path)
        if not payload:
            continue
        try:
            rows = int(payload.get("rows", 0) or 0)
        except Exception:
            rows = 0
        try:
            cols = int(payload.get("cols", 0) or 0)
        except Exception:
            cols = 0
        try:
            header_rows = int(payload.get("header_rows", 0) or 0)
        except Exception:
            header_rows = 0
        cells = payload.get("cells", [])
        if not isinstance(cells, list):
            cells = []
        out[table_id] = {
            "table_id": table_id,
            "model": str(payload.get("model", "") or "").strip(),
            "rows": rows,
            "cols": cols,
            "header_rows": header_rows,
            "cells": [dict(item) for item in cells if isinstance(item, dict)],
            "crop_path": str(payload.get("crop_path", "")).strip(),
            "html_table": str(payload.get("html_table", "") or "").strip(),
        }
    return out


def _read_ocr_evidence(
    doc_dir: Path,
    table_payload: dict[str, Any],
    page: int,
    fragment_rows: list[dict[str, Any]],
) -> str:
    snippets: list[str] = []
    seen_paths: set[str] = set()

    def _append_if_exists(path: Path) -> None:
        key = str(path)
        if key in seen_paths or not path.exists() or not path.is_file():
            return
        try:
            text = path.read_text()
        except Exception:
            return
        seen_paths.add(key)
        snippets.append(f"[{path.name}]\n{text}")

    ocr_merge = table_payload.get("ocr_merge", {})
    if isinstance(ocr_merge, dict):
        rel_path = str(ocr_merge.get("ocr_html_path", "")).strip()
        if rel_path:
            _append_if_exists(doc_dir / rel_path)
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"
    if qa_root.exists():
        for frag in fragment_rows:
            if not isinstance(frag, dict):
                continue
            group = str(frag.get("table_group_id", "")).strip()
            m = GROUP_PAGE_TABLE_RE.search(group)
            if not m:
                continue
            frag_page = int(m.group(1))
            ordinal = int(m.group(2))
            _append_if_exists(qa_root / f"table_{ordinal:02d}_page_{frag_page:04d}.md")
        pages = table_payload.get("pages", [])
        if not isinstance(pages, list):
            pages = []
        candidate_pages = {int(page)}
        for item in pages:
            try:
                p = int(item)
            except Exception:
                continue
            if p > 0:
                candidate_pages.add(p)
        for p in sorted(candidate_pages):
            for path in sorted(qa_root.glob(f"table_*_page_{p:04d}.md"))[:6]:
                _append_if_exists(path)
    return "\n\n".join(snippets)


def _collect_fragment_rows(
    *,
    table_payload: dict[str, Any],
    fragment_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fragment_ids = table_payload.get("fragment_ids", [])
    if not isinstance(fragment_ids, list):
        fragment_ids = []
    for item in fragment_ids:
        fid = str(item).strip()
        if not fid:
            continue
        frag = fragment_index.get(fid)
        if isinstance(frag, dict):
            rows.append(frag)
    if rows:
        return rows
    row_lineage = table_payload.get("row_lineage", [])
    if isinstance(row_lineage, list):
        seen_fids: set[str] = set()
        for item in row_lineage:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("fragment_id", "")).strip()
            if not fid or fid in seen_fids:
                continue
            seen_fids.add(fid)
            frag = fragment_index.get(fid)
            if isinstance(frag, dict):
                rows.append(frag)
    return rows


def _collect_page_context_text(
    *,
    doc_dir: Path,
    table_payload: dict[str, Any],
    page: int,
    context_window: str,
) -> str:
    window = str(context_window or "none").strip().lower()
    if window not in CONTEXT_WINDOW_OPTIONS or window == "none":
        return ""

    base_pages: set[int] = set()
    try:
        current_page = int(page)
    except Exception:
        current_page = 0
    if current_page > 0:
        base_pages.add(current_page)
    raw_pages = table_payload.get("pages", [])
    if isinstance(raw_pages, list):
        for item in raw_pages:
            try:
                p = int(item)
            except Exception:
                continue
            if p > 0:
                base_pages.add(p)
    if not base_pages:
        return ""

    deltas = (0,)
    if window == "page_plus_minus_1":
        deltas = (-1, 0, 1)
    elif window == "page_plus_minus_2":
        deltas = (-2, -1, 0, 1, 2)

    target_pages: set[int] = set()
    for p in base_pages:
        for delta in deltas:
            candidate = p + delta
            if candidate > 0:
                target_pages.add(candidate)

    snippets: list[str] = []
    pages_dir = doc_dir / "pages"
    for candidate_page in sorted(target_pages):
        page_path = pages_dir / f"{int(candidate_page):04d}.md"
        if not page_path.exists():
            continue
        try:
            snippets.append(f"[page_{candidate_page:04d}.md]\n{page_path.read_text()}")
        except Exception:
            continue
    return "\n\n".join(snippets)


def _collect_evidence_bundle(
    doc_dir: Path,
    table_payload: dict[str, Any],
    page: int,
    *,
    fragment_index: dict[str, dict[str, Any]],
    table_structure: dict[str, Any],
    context_window: str = "none",
) -> dict[str, str]:
    fragment_rows = _collect_fragment_rows(table_payload=table_payload, fragment_index=fragment_index)
    context_text = _collect_page_context_text(
        doc_dir=doc_dir,
        table_payload=table_payload,
        page=page,
        context_window=context_window,
    )
    headers = [str(x) for row in table_payload.get("header_rows_full", []) if isinstance(row, list) for x in row]
    rows = [str(x) for row in table_payload.get("data_rows", []) if isinstance(row, list) for x in row]
    fragment_text_parts: list[str] = []
    for frag in fragment_rows:
        header_rows = frag.get("header_rows", [])
        data_rows = frag.get("data_rows", [])
        frag_caption = str(frag.get("caption_text", "")).strip()
        frag_page = int(frag.get("page", 0) or 0)
        frag_group = str(frag.get("table_group_id", "")).strip()
        header_blob = json.dumps(header_rows, ensure_ascii=True) if isinstance(header_rows, list) else "[]"
        data_blob = json.dumps(data_rows, ensure_ascii=True) if isinstance(data_rows, list) else "[]"
        fragment_text_parts.append(
            f"[fragment page={frag_page} group={frag_group} caption={frag_caption}] headers={header_blob} rows={data_blob}"
        )
    marker_fragment_text = "\n".join(fragment_text_parts)
    marker_text = "\n".join(headers + rows + [str(table_payload.get("caption_text", "")), marker_fragment_text])
    ocr_text = _read_ocr_evidence(doc_dir, table_payload, page, fragment_rows)
    return {
        "context_text": context_text,
        "marker_text": marker_text,
        "marker_fragment_text": marker_fragment_text,
        "ocr_text": ocr_text,
        "table_structure_text": json.dumps(table_structure, ensure_ascii=True) if table_structure else "",
        "combined_text": "\n".join([marker_text, ocr_text, context_text]),
    }


def _build_prompt(
    *,
    table_payload: dict[str, Any],
    manifest_row: dict[str, Any],
    risk_score: float,
    evidence: dict[str, str],
    validation_feedback: dict[str, Any],
    table_structure: dict[str, Any],
    context_window: str = "none",
    context_mode: str = "on_demand",
) -> str:
    active_context = evidence.get("context_text", "")
    request = {
        "table_id": str(table_payload.get("table_id", manifest_row.get("table_id", ""))),
        "page": int(manifest_row.get("page", 0) or 0),
        "risk_score": risk_score,
        "context_mode": str(context_mode),
        "context_window_active": str(context_window),
        "canonical_table": {
            "caption": str(table_payload.get("caption_text", manifest_row.get("caption", ""))),
            "header_rows_full": table_payload.get("header_rows_full", []),
            "data_rows": table_payload.get("data_rows", []),
            "row_lineage": table_payload.get("row_lineage", []),
            "context_mappings": table_payload.get("context_mappings", []),
            "required_fields_missing": table_payload.get("required_fields_missing", []),
            "quality_metrics": table_payload.get("quality_metrics", {}),
        },
        "validation_feedback": validation_feedback,
        "table_structure": table_structure,
        "ocr_table_raw": evidence["ocr_text"][:24000],
        "context_tool": {
            "available": True,
            "allowed_windows": list(CONTEXT_WINDOW_OPTIONS),
            "use_only_if_needed": True,
            "examples": [
                "aliases_or_abbreviations_without_definition",
                "units_or_material_names_missing_from_table_body",
                "caption_or_body_text_needed_to_disambiguate_columns",
            ],
        },
        "evidence": {
            "marker_and_caption": evidence["marker_text"][:18000],
            "marker_fragment_rows": evidence.get("marker_fragment_text", "")[:22000],
            "nearby_pages_text": active_context[:24000],
            "table_structure_raw": evidence.get("table_structure_text", "")[:18000],
        },
    }
    return (
        "You are a table rectification model.\n"
        "Return JSON only.\n"
        "Rules:\n"
        "- Evidence-bounded: do not invent values not supported by marker, OCR, caption, or provided context evidence.\n"
        "- Preserve units/symbols and table semantics.\n"
        "- Prioritize column alignment, multi-level header integrity, and row-to-column correctness over preserving superficial layout.\n"
        "- Structure lock: when table_structure is provided, preserve its row/column topology and spans.\n"
        "- If values are vertically stacked across lines, merge them into their logical cell.\n"
        "- If previous validation feedback lists specific failure modes or instructions, address them explicitly.\n"
        "- Context discipline: default to table-local evidence (marker+caption+OCR). Request page context only when strictly needed.\n"
        "- Keep output rectangular.\n"
        "- Every non-empty changed cell must include provenance.\n"
        "Output schema keys (all required):\n"
        "rectified_header_rows_full, rectified_rows, edits, cell_provenance, rectifier_confidence, needs_review.\n\n"
        "Optional key:\n"
        "context_request with fields needed (bool), window (none|table_page|page_plus_minus_1|page_plus_minus_2), reason (string).\n\n"
        f"Input:\n{json.dumps(request, ensure_ascii=True)}"
    )


def _build_repair_prompt(raw_text: str) -> str:
    return (
        "Repair this malformed model output into one strict JSON object.\n"
        "Output JSON only.\n"
        "Required keys: rectified_header_rows_full, rectified_rows, edits, cell_provenance, rectifier_confidence, needs_review.\n\n"
        f"Malformed output:\n{raw_text[:14000]}"
    )


def _build_schema_retry_prompt(*, schema_reason: str, raw_payload: dict[str, Any], table_payload: dict[str, Any]) -> str:
    request = {
        "schema_reason": schema_reason,
        "required_keys": [
            "rectified_header_rows_full",
            "rectified_rows",
            "edits",
            "cell_provenance",
            "rectifier_confidence",
            "needs_review",
        ],
        "minimum_requirements": {
            "nonempty_rectified_header_rows_full": True,
            "nonempty_rectified_rows": True,
            "rectangular_grid": True,
            "include_cell_provenance_for_changed_cells": True,
        },
        "fallback_if_unsure": {
            "rectified_header_rows_full": table_payload.get("header_rows_full", []),
            "rectified_rows": table_payload.get("data_rows", []),
        },
        "previous_output": raw_payload,
    }
    return "Your previous rectification output was invalid. Return corrected JSON only.\n" + json.dumps(request, ensure_ascii=True)


def _build_evidence_retry_prompt(
    *,
    evidence_reason: str,
    previous_output: dict[str, Any],
    table_payload: dict[str, Any],
    manifest_row: dict[str, Any],
    evidence: dict[str, str],
    validation_feedback: dict[str, Any],
    table_structure: dict[str, Any],
) -> str:
    request = {
        "evidence_violation": evidence_reason,
        "table_id": str(table_payload.get("table_id", manifest_row.get("table_id", ""))),
        "expected_fixes": validation_feedback,
        "table_structure": table_structure,
        "previous_output": previous_output,
        "source_table": {
            "caption": str(table_payload.get("caption_text", manifest_row.get("caption", ""))),
            "header_rows_full": table_payload.get("header_rows_full", []),
            "data_rows": table_payload.get("data_rows", []),
        },
        "evidence": {
            "marker_and_caption": evidence.get("marker_text", "")[:18000],
            "marker_fragment_rows": evidence.get("marker_fragment_text", "")[:22000],
            "ocr_table_raw": evidence.get("ocr_text", "")[:24000],
            "nearby_pages_text": evidence.get("context_text", "")[:24000],
            "table_structure_raw": evidence.get("table_structure_text", "")[:18000],
        },
    }
    return (
        "Your previous rectification failed evidence validation.\n"
        "Return corrected JSON only.\n"
        "Focus on: preserving headers/units/symbols, fixing row/column alignment, and grounding every changed cell in evidence.\n"
        "Do not hallucinate values.\n\n"
        f"Input:\n{json.dumps(request, ensure_ascii=True)}"
    )


def _extract_context_request(payload: dict[str, Any]) -> tuple[bool, str, str]:
    raw = payload.get("context_request")
    if not isinstance(raw, dict):
        return False, "none", ""
    needed = bool(raw.get("needed", False))
    window = str(raw.get("window", "none") or "none").strip().lower()
    if window not in CONTEXT_WINDOW_OPTIONS:
        window = "none"
    reason = _normalize_cell_text(raw.get("reason", ""))
    if not needed or window == "none":
        return False, "none", reason
    return True, window, reason


def _normalize_rectified_payload(payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    header_rows_raw = payload.get("rectified_header_rows_full")
    rows_raw = payload.get("rectified_rows")
    edits_raw = payload.get("edits")
    provenance_raw = payload.get("cell_provenance")
    if not isinstance(header_rows_raw, list) or not isinstance(rows_raw, list):
        return False, "invalid_schema:missing_grid", {}
    if not isinstance(edits_raw, list):
        return False, "invalid_schema:missing_edits", {}
    if not isinstance(provenance_raw, (list, dict)):
        return False, "invalid_schema:missing_cell_provenance", {}

    header_rows: list[list[str]] = []
    for row in header_rows_raw:
        if not isinstance(row, list):
            return False, "invalid_schema:header_row_not_list", {}
        header_rows.append([_normalize_cell_text(cell) for cell in row])

    rectified_rows: list[list[str]] = []
    for row in rows_raw:
        if not isinstance(row, list):
            return False, "invalid_schema:data_row_not_list", {}
        rectified_rows.append([_normalize_cell_text(cell) for cell in row])

    target_cols = max(
        max((len(row) for row in header_rows), default=0),
        max((len(row) for row in rectified_rows), default=0),
    )
    if target_cols <= 0:
        return False, "invalid_schema:empty_grid", {}
    normalized_headers = [list(row) + [""] * (target_cols - len(row)) for row in header_rows]
    normalized_rows = [list(row) + [""] * (target_cols - len(row)) for row in rectified_rows]
    normalized_rows = [row[:target_cols] for row in normalized_rows]
    normalized_headers = [row[:target_cols] for row in normalized_headers]

    rectifier_confidence = payload.get("rectifier_confidence", 0.0)
    try:
        confidence = float(rectifier_confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    needs_review = bool(payload.get("needs_review", True))
    edits = [dict(item) for item in edits_raw if isinstance(item, dict)]
    return (
        True,
        "",
        {
            "header_rows_full": normalized_headers,
            "rows": normalized_rows,
            "headers": _collapse_header_rows(normalized_headers),
            "edits": edits,
            "cell_provenance_raw": provenance_raw,
            "rectifier_confidence": confidence,
            "needs_review": needs_review,
        },
    )


def _normalized_noop_from_table_payload(table_payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    header_rows_raw = table_payload.get("header_rows_full", [])
    data_rows_raw = table_payload.get("data_rows", [])
    header_rows: list[list[str]] = []
    for row in header_rows_raw if isinstance(header_rows_raw, list) else []:
        if isinstance(row, list):
            header_rows.append([_normalize_cell_text(cell) for cell in row])
    data_rows: list[list[str]] = []
    for row in data_rows_raw if isinstance(data_rows_raw, list) else []:
        if isinstance(row, list):
            data_rows.append([_normalize_cell_text(cell) for cell in row])
    target_cols = max(
        max((len(row) for row in header_rows), default=0),
        max((len(row) for row in data_rows), default=0),
    )
    if target_cols <= 0:
        return False, "invalid_schema:empty_grid", {}
    if not header_rows:
        header_rows = [[""] * target_cols]
    normalized_headers = [(list(row) + [""] * (target_cols - len(row)))[:target_cols] for row in header_rows]
    normalized_rows = [(list(row) + [""] * (target_cols - len(row)))[:target_cols] for row in data_rows]
    return (
        True,
        "",
        {
            "header_rows_full": normalized_headers,
            "rows": normalized_rows,
            "headers": _collapse_header_rows(normalized_headers),
            "edits": [
                {
                    "type": "schema_fallback_noop",
                    "description": "LLM output invalid; retained deterministic extraction.",
                }
            ],
            "cell_provenance_raw": table_payload.get("cell_provenance", []),
            "rectifier_confidence": 0.0,
            "needs_review": True,
        },
    )


def _normalize_provenance(raw: Any) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        candidates = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            m = re.match(r"^\s*(\d+)\s*[:,]\s*(\d+)\s*$", str(key))
            if not m:
                continue
            item = dict(value)
            item["row"] = int(m.group(1))
            item["col"] = int(m.group(2))
            candidates.append(item)
    else:
        candidates = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            row = int(item.get("row"))
            col = int(item.get("col"))
        except Exception:
            continue
        source = str(item.get("source", "")).strip().lower()
        evidence_text = _normalize_cell_text(item.get("evidence_text", ""))
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        out[(row, col)] = {
            "row": row,
            "col": col,
            "source": source,
            "evidence_text": evidence_text,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return out


def _looks_substantive(value: str) -> bool:
    text = _normalize_cell_text(value)
    return bool(text and text not in {"-", "—", "–", "na", "n/a"})


def _infer_provenance_for_changed_cells(
    *,
    provenance_by_cell: dict[tuple[int, int], dict[str, Any]],
    rows: list[list[str]],
    orig_rows: list[list[str]],
    evidence: dict[str, str],
) -> dict[tuple[int, int], dict[str, Any]]:
    out = dict(provenance_by_cell)
    marker_text = str(evidence.get("marker_text", ""))
    ocr_text = str(evidence.get("ocr_text", ""))
    context_text = str(evidence.get("context_text", ""))
    marker_lower = marker_text.lower()
    ocr_lower = ocr_text.lower()
    context_lower = context_text.lower()

    def _pick_source(cell_text: str, old_text: str) -> tuple[str, str, float] | None:
        cell_lower = cell_text.lower()
        if cell_lower and cell_lower in marker_lower:
            return ("marker", cell_text, 0.86)
        if cell_lower and cell_lower in ocr_lower:
            return ("ocr", cell_text, 0.81)
        if cell_lower and cell_lower in context_lower:
            return ("context", cell_text, 0.72)
        old_lower = old_text.lower()
        if old_text and old_lower and old_lower in marker_lower:
            return ("marker", old_text, 0.66)
        return None

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            if (row_idx, col_idx) in out:
                continue
            new_text = _normalize_cell_text(cell)
            old_text = ""
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_text = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if not new_text or new_text == old_text:
                continue
            inferred = _pick_source(new_text, old_text)
            if inferred is None:
                continue
            source, evidence_text, confidence = inferred
            out[(row_idx, col_idx)] = {
                "row": row_idx,
                "col": col_idx,
                "source": source,
                "evidence_text": evidence_text,
                "confidence": confidence,
            }
    return out


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(str(text or "").lower()) if token}


def _canonicalize_symbol(ch: str) -> str:
    dash_chars = {"-", "−", "–", "—", "‑"}
    degree_chars = {"°", "º", "◦"}
    star_chars = {"∗", "＊", "✱", "✳", "✻", "﹡", "*"}
    if ch in dash_chars:
        return "-"
    if ch in degree_chars:
        return "°"
    if ch in star_chars:
        return "*"
    return ch


def _canonical_symbol_set(text: str) -> set[str]:
    symbols = {_canonicalize_symbol(sym) for sym in _extract_symbols(text)}
    return {sym for sym in symbols if sym not in IGNORABLE_SYMBOLS}


def _alnum_signature(text: str) -> str:
    return "".join(token for token in TOKEN_RE.findall(str(text or "").lower()) if token)


def _text_supported_by_source(evidence_text: str, source_blob: str) -> bool:
    needle = _normalize_cell_text(evidence_text)
    hay = _normalize_cell_text(source_blob)
    if not needle:
        return False
    if needle.lower() in hay.lower():
        return True
    compact_needle = re.sub(r"\s+", "", needle.lower())
    compact_hay = re.sub(r"\s+", "", hay.lower())
    compact_needle = compact_needle.replace("−", "-").replace("–", "-").replace("—", "-")
    compact_hay = compact_hay.replace("−", "-").replace("–", "-").replace("—", "-")
    if compact_needle and NUMERICISH_RE.match(compact_needle):
        return compact_needle in compact_hay
    n_sig = _alnum_signature(needle)
    h_sig = _alnum_signature(hay)
    alpha_chars = sum(1 for ch in n_sig if "a" <= ch <= "z")
    if n_sig and n_sig in h_sig and (len(n_sig) >= 6 or alpha_chars >= 3):
        return True
    n_tokens = _tokenize(needle)
    h_tokens = _tokenize(hay)
    has_alpha_token = any(any("a" <= ch <= "z" for ch in token) for token in n_tokens)
    if n_tokens and h_tokens and has_alpha_token:
        overlap = len(n_tokens.intersection(h_tokens)) / max(len(n_tokens), 1)
        if overlap >= 0.7:
            return True
    return False


def _auto_repair_candidate_for_validation(
    *,
    headers: list[str],
    rows: list[list[str]],
    orig_headers: list[str],
    orig_rows: list[list[str]],
    provenance_by_cell: dict[tuple[int, int], dict[str, Any]],
    evidence: dict[str, str],
) -> tuple[list[str], list[list[str]], dict[tuple[int, int], dict[str, Any]]]:
    repaired_headers = [str(x) for x in headers]
    repaired_rows = [[str(cell) for cell in row] for row in rows]
    repaired_provenance = dict(provenance_by_cell)

    # If the model drops too much structure, retain deterministic structure.
    orig_nonempty_headers = sum(1 for h in orig_headers if _normalize_cell_text(h))
    new_nonempty_headers = sum(1 for h in repaired_headers if _normalize_cell_text(h))
    if orig_nonempty_headers > 0 and new_nonempty_headers < max(1, int(0.6 * orig_nonempty_headers)):
        repaired_headers = list(orig_headers)
    if orig_rows and len(repaired_rows) < len(orig_rows):
        repaired_rows = [list(row) for row in orig_rows]

    target_cols = max(
        len(repaired_headers),
        max((len(row) for row in repaired_rows), default=0),
        max((len(row) for row in orig_rows), default=0),
    )
    if target_cols > 0:
        repaired_headers = (repaired_headers + [""] * (target_cols - len(repaired_headers)))[:target_cols]
        normalized_rows: list[list[str]] = []
        for row in repaired_rows:
            normalized_rows.append((list(row) + [""] * (target_cols - len(row)))[:target_cols])
        repaired_rows = normalized_rows
        normalized_orig_rows: list[list[str]] = []
        for row in orig_rows:
            normalized_orig_rows.append((list(row) + [""] * (target_cols - len(row)))[:target_cols])
        orig_rows = normalized_orig_rows
        orig_headers = (list(orig_headers) + [""] * (target_cols - len(orig_headers)))[:target_cols]

    # Revert per-cell changes that would drop non-ignorable symbols.
    for col_idx in range(min(len(orig_headers), len(repaired_headers))):
        old_cell = _normalize_cell_text(orig_headers[col_idx])
        new_cell = _normalize_cell_text(repaired_headers[col_idx])
        old_has_unit = bool(UNIT_TOKEN_RE.search(old_cell))
        new_has_unit = bool(UNIT_TOKEN_RE.search(new_cell))
        if old_has_unit and not new_has_unit:
            repaired_headers[col_idx] = old_cell
            continue
        if old_cell and _canonical_symbol_set(old_cell).difference(_canonical_symbol_set(new_cell)):
            repaired_headers[col_idx] = old_cell
    for row_idx in range(min(len(orig_rows), len(repaired_rows))):
        for col_idx in range(min(len(orig_rows[row_idx]), len(repaired_rows[row_idx]))):
            old_cell = _normalize_cell_text(orig_rows[row_idx][col_idx])
            new_cell = _normalize_cell_text(repaired_rows[row_idx][col_idx])
            if old_cell and _canonical_symbol_set(old_cell).difference(_canonical_symbol_set(new_cell)):
                repaired_rows[row_idx][col_idx] = old_cell

    marker_text = str(evidence.get("marker_text", ""))
    ocr_text = str(evidence.get("ocr_text", ""))
    context_text = str(evidence.get("context_text", ""))
    # Backfill provenance for changed substantive cells to reduce unnecessary rejections.
    for row_idx, row in enumerate(repaired_rows):
        for col_idx, cell in enumerate(row):
            new_value = _normalize_cell_text(cell)
            old_value = ""
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_value = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if not _looks_substantive(new_value) or new_value == old_value:
                continue
            key = (row_idx, col_idx)
            if key in repaired_provenance:
                continue
            source = "marker"
            evidence_text = old_value or new_value
            confidence = 0.62
            if _text_supported_by_source(new_value, ocr_text):
                source = "ocr"
                evidence_text = new_value
                confidence = 0.78
            elif _text_supported_by_source(new_value, marker_text):
                source = "marker"
                evidence_text = new_value
                confidence = 0.82
            elif _text_supported_by_source(new_value, context_text):
                source = "context"
                evidence_text = new_value
                confidence = 0.7
            elif _text_supported_by_source(old_value, marker_text):
                source = "marker"
                evidence_text = old_value
                confidence = 0.66
            repaired_provenance[key] = {
                "row": row_idx,
                "col": col_idx,
                "source": source,
                "evidence_text": evidence_text,
                "confidence": confidence,
            }

    return repaired_headers, repaired_rows, repaired_provenance


def _validate_evidence(
    *,
    table_payload: dict[str, Any],
    normalized: dict[str, Any],
    evidence: dict[str, str],
    table_structure: dict[str, Any],
    structure_lock: bool,
) -> tuple[bool, str, list[dict[str, Any]], list[str], list[list[str]]]:
    headers = [str(x) for x in normalized.get("headers", [])]
    rows = [[str(x) for x in row] for row in normalized.get("rows", []) if isinstance(row, list)]
    if not rows:
        return False, "evidence_violation:no_rows", [], headers, rows

    orig_header_rows = table_payload.get("header_rows_full", [])
    orig_headers = _collapse_header_rows(
        [[_normalize_cell_text(cell) for cell in row] for row in orig_header_rows if isinstance(row, list)]
    )
    orig_rows = [
        [_normalize_cell_text(cell) for cell in row]
        for row in table_payload.get("data_rows", [])
        if isinstance(row, list)
    ]
    if not headers and orig_headers:
        headers = list(orig_headers)
    target_cols = max(
        len(headers),
        max((len(row) for row in rows), default=0),
    )
    if target_cols <= 0:
        return False, "invalid_schema:no_headers", [], headers, rows
    headers = (headers + [""] * (target_cols - len(headers)))[:target_cols]
    rows = [(row + [""] * (target_cols - len(row)))[:target_cols] for row in rows]

    if structure_lock and isinstance(table_structure, dict):
        expected_cols = 0
        expected_rows = 0
        expected_header_rows = 0
        try:
            expected_cols = int(table_structure.get("cols", 0) or 0)
        except Exception:
            expected_cols = 0
        try:
            expected_rows = int(table_structure.get("rows", 0) or 0)
        except Exception:
            expected_rows = 0
        try:
            expected_header_rows = int(table_structure.get("header_rows", 0) or 0)
        except Exception:
            expected_header_rows = 0

        candidate_header_rows_full = _coerce_header_rows_full(normalized.get("header_rows_full", []))
        candidate_total_rows = len(candidate_header_rows_full) + len(rows)
        if expected_cols > 0 and target_cols != expected_cols:
            return False, "evidence_violation:structure_col_mismatch", [], headers, rows
        if expected_rows > 0 and candidate_total_rows != expected_rows:
            return False, "evidence_violation:structure_row_mismatch", [], headers, rows
        if expected_header_rows > 0 and len(candidate_header_rows_full) != expected_header_rows:
            return False, "evidence_violation:structure_header_mismatch", [], headers, rows

    orig_nonempty_headers = sum(1 for h in orig_headers if _normalize_cell_text(h))
    new_nonempty_headers = sum(1 for h in headers if _normalize_cell_text(h))
    provenance_by_cell = _normalize_provenance(normalized.get("cell_provenance_raw"))
    provenance_by_cell = _infer_provenance_for_changed_cells(
        provenance_by_cell=provenance_by_cell,
        rows=rows,
        orig_rows=orig_rows,
        evidence=evidence,
    )
    headers, rows, provenance_by_cell = _auto_repair_candidate_for_validation(
        headers=headers,
        rows=rows,
        orig_headers=orig_headers,
        orig_rows=orig_rows,
        provenance_by_cell=provenance_by_cell,
        evidence=evidence,
    )
    orig_nonempty_headers = sum(1 for h in orig_headers if _normalize_cell_text(h))
    new_nonempty_headers = sum(1 for h in headers if _normalize_cell_text(h))
    if orig_nonempty_headers > 0 and new_nonempty_headers < max(1, int(0.6 * orig_nonempty_headers)):
        return False, "evidence_violation:header_loss", [], headers, rows
    if orig_rows and len(rows) < max(1, int(0.5 * len(orig_rows))):
        return False, "evidence_violation:row_loss", [], headers, rows

    normalized_provenance = sorted(provenance_by_cell.values(), key=lambda row: (int(row["row"]), int(row["col"])))
    marker_lower = evidence.get("marker_text", "").lower()
    ocr_lower = evidence.get("ocr_text", "").lower()
    context_lower = evidence.get("context_text", "").lower()
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            new_value = _normalize_cell_text(cell)
            old_value = ""
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_value = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if _looks_substantive(new_value) and new_value != old_value:
                provenance = provenance_by_cell.get((row_idx, col_idx))
                if provenance is None:
                    return False, f"evidence_violation:missing_provenance:{row_idx}:{col_idx}", [], headers, rows
    for key, provenance in list(provenance_by_cell.items()):
        source = str(provenance.get("source", "")).lower()
        evidence_text = str(provenance.get("evidence_text", "")).strip()
        confidence = float(provenance.get("confidence", 0.0) or 0.0)
        if source not in PROVENANCE_SOURCES:
            row_idx, col_idx = key
            new_text = ""
            old_text = ""
            if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                new_text = _normalize_cell_text(rows[row_idx][col_idx])
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_text = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if (not _looks_substantive(new_text)) or new_text == old_text:
                provenance_by_cell.pop(key, None)
                continue
            candidates = [evidence_text, new_text, old_text]
            source = ""
            for candidate in candidates:
                if candidate and _text_supported_by_source(candidate, marker_lower):
                    source = "marker"
                    evidence_text = candidate
                    break
                if candidate and _text_supported_by_source(candidate, ocr_lower):
                    source = "ocr"
                    evidence_text = candidate
                    break
                if candidate and _text_supported_by_source(candidate, context_lower):
                    source = "context"
                    evidence_text = candidate
                    break
            if source not in PROVENANCE_SOURCES:
                return False, f"evidence_violation:bad_source:{key[0]}:{key[1]}", [], headers, rows
            provenance["source"] = source
            provenance["evidence_text"] = evidence_text
        if not evidence_text:
            return False, f"evidence_violation:missing_evidence_text:{key[0]}:{key[1]}", [], headers, rows
        evidence_text_lower = evidence_text.lower()
        source_blob = marker_lower if source == "marker" else (ocr_lower if source == "ocr" else context_lower)
        if source == "context" and not source_blob:
            row_idx, col_idx = key
            new_text = ""
            old_text = ""
            if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                new_text = _normalize_cell_text(rows[row_idx][col_idx])
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_text = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if _looks_substantive(new_text) and new_text != old_text:
                return False, f"evidence_violation:context_not_loaded:{key[0]}:{key[1]}", [], headers, rows
        if source_blob and (evidence_text_lower not in source_blob) and (not _text_supported_by_source(evidence_text, source_blob)):
            return False, f"evidence_violation:provenance_mismatch:{key[0]}:{key[1]}", [], headers, rows
        if confidence <= 0.0:
            row_idx, col_idx = key
            new_text = ""
            old_text = ""
            if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                new_text = _normalize_cell_text(rows[row_idx][col_idx])
            if row_idx < len(orig_rows) and col_idx < len(orig_rows[row_idx]):
                old_text = _normalize_cell_text(orig_rows[row_idx][col_idx])
            if (not _looks_substantive(new_text)) or new_text == old_text:
                provenance_by_cell.pop(key, None)
                continue
            provenance["confidence"] = 0.55

    original_union_symbols = _canonical_symbol_set(
        "\n".join(
            [
                " | ".join(orig_headers),
                "\n".join(" | ".join(row) for row in orig_rows),
                evidence.get("ocr_text", ""),
            ]
        )
    )
    rectified_symbols = _canonical_symbol_set(
        "\n".join([" | ".join(headers), "\n".join(" | ".join(row) for row in rows)])
    )
    missing_symbols_set = {sym for sym in original_union_symbols if sym not in rectified_symbols}
    original_table_symbols = _canonical_symbol_set(
        "\n".join(
            [
                " | ".join(orig_headers),
                "\n".join(" | ".join(row) for row in orig_rows),
            ]
        )
    )
    missing_symbols = "".join(sorted(missing_symbols_set))
    if missing_symbols:
        # OCR-only symbols are informative but should not hard-fail otherwise valid rectifications.
        if missing_symbols_set.intersection(original_table_symbols):
            return False, f"evidence_violation:symbol_loss:{missing_symbols}", [], headers, rows

    header_text = " ".join(headers)
    original_units = bool(UNIT_TOKEN_RE.search(" ".join(orig_headers + [_normalize_cell_text(table_payload.get("caption_text", ""))])))
    rectified_units = bool(UNIT_TOKEN_RE.search(header_text))
    if original_units and not rectified_units:
        return False, "evidence_violation:unit_loss", [], headers, rows

    evidence_tokens = _tokenize(evidence.get("combined_text", ""))
    rectified_tokens = _tokenize(" ".join(headers + [cell for row in rows for cell in row]))
    novel_tokens = rectified_tokens.difference(evidence_tokens)
    if novel_tokens and len(novel_tokens) > max(3, int(0.2 * max(len(rectified_tokens), 1))):
        return False, "evidence_violation:hallucinated_tokens", [], headers, rows

    return True, "", normalized_provenance, headers, rows


def _write_table_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([_normalize_cell_text(h) for h in headers])
        for row in rows:
            writer.writerow([_normalize_cell_text(cell) for cell in row])


def _coerce_header_rows_full(value: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(value, list):
        return out
    for row in value:
        if not isinstance(row, list):
            continue
        out.append([_normalize_cell_text(cell) for cell in row])
    return out


def _normalize_header_rows_shape(header_rows_full: list[list[str]], target_cols: int) -> list[list[str]]:
    if target_cols <= 0:
        return []
    out: list[list[str]] = []
    for row in header_rows_full:
        out.append((list(row) + [""] * (target_cols - len(row)))[:target_cols])
    return out


def _resolve_header_rows_full_for_apply(
    *,
    candidate_header_rows_full: Any,
    fallback_header_rows_full: Any,
    headers: list[str],
) -> list[list[str]]:
    target_cols = len(headers)
    if target_cols <= 0:
        return []
    candidate = _normalize_header_rows_shape(_coerce_header_rows_full(candidate_header_rows_full), target_cols)
    fallback = _normalize_header_rows_shape(_coerce_header_rows_full(fallback_header_rows_full), target_cols)
    if candidate and _collapse_header_rows(candidate) == headers:
        return candidate
    if fallback and _collapse_header_rows(fallback) == headers:
        return fallback
    return [list(headers)]


def _apply_rectification(
    *,
    doc_dir: Path,
    table_payload: dict[str, Any],
    normalized: dict[str, Any],
    risk_score: float,
    config: RectifierConfig,
) -> tuple[dict[str, Any], list[str], list[list[str]]]:
    headers = [str(x) for x in normalized.get("headers", [])]
    rows = [[str(x) for x in row] for row in normalized.get("rows", []) if isinstance(row, list)]
    header_rows_full = [[str(x) for x in row] for row in normalized.get("header_rows_full", []) if isinstance(row, list)]
    table_id = str(table_payload.get("table_id", ""))

    snapshot = {
        "header_rows_full": table_payload.get("header_rows_full", []),
        "data_rows": table_payload.get("data_rows", []),
        "header_hierarchy": table_payload.get("header_hierarchy", []),
        "row_lineage": table_payload.get("row_lineage", []),
        "context_mappings": table_payload.get("context_mappings", []),
        "required_fields_missing": table_payload.get("required_fields_missing", []),
        "quality_metrics": table_payload.get("quality_metrics", {}),
    }

    updated = dict(table_payload)
    updated["header_rows"] = [headers]
    updated["header_rows_full"] = header_rows_full
    updated["header_hierarchy"] = [{"level": idx + 1, "cells": row} for idx, row in enumerate(header_rows_full)]
    updated["data_rows"] = rows
    updated["quality_metrics"] = _quality_metrics(headers, rows)
    updated["cell_provenance"] = normalized.get("cell_provenance", [])
    updated["rectifier_confidence"] = float(normalized.get("rectifier_confidence", 0.0) or 0.0)
    updated["rectifier_needs_review"] = bool(normalized.get("needs_review", True))
    updated["llm_rectification"] = {
        "applied": True,
        "model": config.model,
        "target": config.target,
        "risk_score": risk_score,
        "edits": [dict(item) for item in normalized.get("edits", []) if isinstance(item, dict)],
        "original_snapshot": snapshot,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    csv_rel = str(table_payload.get("csv_path", "")).strip()
    if not csv_rel:
        csv_rel = f"metadata/assets/structured/extracted/tables/{table_id}.csv"
        updated["csv_path"] = csv_rel
    csv_path = doc_dir / csv_rel
    _write_table_csv(csv_path, headers, rows)
    return updated, headers, rows


async def run_table_rectification_for_doc(
    *,
    doc_dir: Path,
    client: AsyncOpenAI,
    config: RectifierConfig,
    call_model: Callable[..., Awaitable[Any]] = call_text_model,
) -> RectifierResult:
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    report_path = qa_root / "table_llm_rectification.json"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = RectifierResult(
        enabled=True,
        model=config.model,
        target=config.target,
        report_path=_portable_path(report_path, doc_dir),
        run_id=run_id,
    )

    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    tables_manifest_path = tables_dir / "manifest.jsonl"
    manifest_rows = _load_jsonl(tables_manifest_path)
    result.considered = len(manifest_rows)

    qa_flags_path = qa_root / "table_flags.jsonl"
    qa_rows = _load_jsonl(qa_flags_path)
    existing_flag_ids = {str(row.get("flag_id", "")) for row in qa_rows if str(row.get("flag_id", ""))}

    validation_path = doc_dir / "metadata" / "assets" / "structured" / "validation" / "gemini_table_review.jsonl"
    target_table_ids = _collect_target_table_ids(validation_path, target=config.target)
    validation_feedback_by_id = _collect_validation_feedback_map(validation_path)
    fragment_index = _resolve_fragment_index(doc_dir)

    table_payloads: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        table_id = str(row.get("table_id", "")).strip()
        if not table_id:
            continue
        table_path = tables_dir / f"{table_id}.json"
        payload = _load_json(table_path)
        if not payload:
            payload = {
                "table_id": table_id,
                "caption_text": str(row.get("caption", "")),
                "header_rows_full": [list(row.get("headers", []))],
                "data_rows": [list(r) for r in row.get("rows", []) if isinstance(r, list)],
                "quality_metrics": _quality_metrics(
                    [str(x) for x in row.get("headers", [])],
                    [[str(x) for x in r] for r in row.get("rows", []) if isinstance(r, list)],
                ),
                "required_fields_missing": [],
                "row_lineage": [],
                "context_mappings": [],
            }
        table_payloads[table_id] = _baseline_payload_for_rerun(payload)

    candidates: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        table_id = str(manifest_row.get("table_id", "")).strip()
        if not table_id:
            continue
        payload = table_payloads.get(table_id, {})
        risk_score = _compute_risk_score(payload, qa_rows)
        base_row = {"table_id": table_id, "page": int(manifest_row.get("page", 0) or 0), "risk_score": risk_score}
        if bool(config.skip_already_rectified):
            llm_meta = payload.get("llm_rectification", {})
            already_rectified = bool(llm_meta.get("applied")) if isinstance(llm_meta, dict) else False
            if already_rectified:
                _append_flag(
                    qa_rows=qa_rows,
                    existing_flag_ids=existing_flag_ids,
                    table_id=table_id,
                    page=int(manifest_row.get("page", 0) or 0),
                    flag_type="llm_rectification_skipped_already_applied",
                    severity="info",
                    details=f"model={str(llm_meta.get('model', '')).strip()}",
                )
                result.table_results.append({**base_row, "status": "skipped_already_rectified"})
                continue
        if config.target == "all":
            candidates.append({**base_row, "manifest_row": manifest_row, "table_payload": payload})
            continue
        if config.target == "nonaccept":
            if table_id in target_table_ids:
                candidates.append({**base_row, "manifest_row": manifest_row, "table_payload": payload})
                continue
            result.table_results.append({**base_row, "status": "skipped_target_filter"})
            continue
        if config.target == "reject":
            if table_id in target_table_ids:
                candidates.append({**base_row, "manifest_row": manifest_row, "table_payload": payload})
                continue
            result.table_results.append({**base_row, "status": "skipped_target_filter"})
            continue
        if risk_score >= float(config.risk_threshold):
            candidates.append({**base_row, "manifest_row": manifest_row, "table_payload": payload})
            continue
        result.skipped_low_risk += 1
        _append_flag(
            qa_rows=qa_rows,
            existing_flag_ids=existing_flag_ids,
            table_id=table_id,
            page=int(manifest_row.get("page", 0) or 0),
            flag_type="llm_rectification_skipped_low_risk",
            severity="info",
            details=f"risk={risk_score:.4f} threshold={float(config.risk_threshold):.4f}",
        )
        result.table_results.append({**base_row, "status": "skipped_low_risk"})

    candidates = sorted(candidates, key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    if config.max_tables_per_doc > 0:
        candidates = candidates[: int(config.max_tables_per_doc)]
    result.selected = len(candidates)

    if not manifest_rows:
        report_path.write_text(json.dumps(result.__dict__, indent=2, ensure_ascii=True))
        _write_jsonl(qa_flags_path, qa_rows)
        return result

    manifest_by_id = {str(row.get("table_id", "")): dict(row) for row in manifest_rows}
    canonical_path = tables_dir / "canonical.jsonl"
    canonical_rows = _load_jsonl(canonical_path)
    canonical_by_id = {str(row.get("table_id", "")): dict(row) for row in canonical_rows}
    table_structure_by_id = _load_table_structure_map(doc_dir)

    for candidate in candidates:
        table_id = str(candidate["table_id"])
        page = int(candidate.get("page", 0) or 0)
        risk_score = float(candidate.get("risk_score", 0.0) or 0.0)
        manifest_row = dict(candidate.get("manifest_row", {}))
        table_payload = dict(candidate.get("table_payload", {}))
        validation_feedback = dict(validation_feedback_by_id.get(table_id, {}))
        table_structure = dict(table_structure_by_id.get(table_id, {}))
        context_window_used = "none"
        context_requested = False
        context_request_reason = ""
        evidence = _collect_evidence_bundle(
            doc_dir,
            table_payload,
            page,
            fragment_index=fragment_index,
            table_structure=table_structure,
            context_window=context_window_used,
        )
        prompt = _build_prompt(
            table_payload=table_payload,
            manifest_row=manifest_row,
            risk_score=risk_score,
            evidence=evidence,
            validation_feedback=validation_feedback,
            table_structure=table_structure,
            context_window=context_window_used,
            context_mode=str(config.context_mode),
        )
        parsed: dict[str, Any] = {}
        parse_status = "ok"
        try:
            response = await call_model(
                client=client,
                model=config.model,
                prompt=prompt,
                max_tokens=int(config.max_tokens),
            )
            parsed = _extract_json_object(getattr(response, "content", ""))
            if not parsed:
                repair = await call_model(
                    client=client,
                    model=config.model,
                    prompt=_build_repair_prompt(getattr(response, "content", "")),
                    max_tokens=int(config.max_tokens),
                )
                parsed = _extract_json_object(getattr(repair, "content", ""))
        except Exception as exc:  # noqa: BLE001
            result.fallbacked += 1
            result.errors += 1
            _append_flag(
                qa_rows=qa_rows,
                existing_flag_ids=existing_flag_ids,
                table_id=table_id,
                page=page,
                flag_type="llm_rectification_failed_fallback",
                severity="warn",
                details=str(exc),
            )
            result.table_results.append(
                {
                    "table_id": table_id,
                    "page": page,
                    "risk_score": risk_score,
                    "status": "failed_fallback",
                    "error": str(exc),
                    "context_requested": context_requested,
                    "context_window": context_window_used,
                }
            )
            continue

        if parsed and str(config.context_mode).strip().lower() == "on_demand":
            requested, requested_window, requested_reason = _extract_context_request(parsed)
            if requested and requested_window != "none":
                context_requested = True
                context_window_used = requested_window
                context_request_reason = requested_reason
                evidence = _collect_evidence_bundle(
                    doc_dir,
                    table_payload,
                    page,
                    fragment_index=fragment_index,
                    table_structure=table_structure,
                    context_window=context_window_used,
                )
                _append_flag(
                    qa_rows=qa_rows,
                    existing_flag_ids=existing_flag_ids,
                    table_id=table_id,
                    page=page,
                    flag_type="llm_rectification_context_requested",
                    severity="info",
                    details=f"window={context_window_used} reason={context_request_reason or 'unspecified'}",
                )
                context_prompt = _build_prompt(
                    table_payload=table_payload,
                    manifest_row=manifest_row,
                    risk_score=risk_score,
                    evidence=evidence,
                    validation_feedback=validation_feedback,
                    table_structure=table_structure,
                    context_window=context_window_used,
                    context_mode=str(config.context_mode),
                )
                try:
                    context_response = await call_model(
                        client=client,
                        model=config.model,
                        prompt=context_prompt,
                        max_tokens=int(config.max_tokens),
                    )
                    context_parsed = _extract_json_object(getattr(context_response, "content", ""))
                    if not context_parsed:
                        context_repair = await call_model(
                            client=client,
                            model=config.model,
                            prompt=_build_repair_prompt(getattr(context_response, "content", "")),
                            max_tokens=int(config.max_tokens),
                        )
                        context_parsed = _extract_json_object(getattr(context_repair, "content", ""))
                    if context_parsed:
                        parsed = context_parsed
                except Exception:
                    pass

        schema_retry_count = 0
        ok_schema, schema_reason, normalized = _normalize_rectified_payload(parsed)
        while (not ok_schema) and schema_retry_count < 2 and schema_reason in {
            "invalid_schema:missing_grid",
            "invalid_schema:empty_grid",
        }:
            schema_retry_count += 1
            try:
                retry = await call_model(
                    client=client,
                    model=config.model,
                    prompt=_build_schema_retry_prompt(
                        schema_reason=schema_reason,
                        raw_payload=parsed,
                        table_payload=table_payload,
                    ),
                    max_tokens=int(config.max_tokens),
                )
            except Exception:
                break
            retry_parsed = _extract_json_object(getattr(retry, "content", ""))
            if not retry_parsed:
                break
            parsed = retry_parsed
            ok_schema, schema_reason, normalized = _normalize_rectified_payload(parsed)
        used_schema_noop_fallback = False
        if (not ok_schema) and schema_reason in {"invalid_schema:missing_grid", "invalid_schema:empty_grid"}:
            ok_noop, _, normalized_noop = _normalized_noop_from_table_payload(table_payload)
            if ok_noop:
                ok_schema = True
                normalized = normalized_noop
                used_schema_noop_fallback = True
        if not ok_schema:
            result.fallbacked += 1
            result.invalid_schema += 1
            _append_flag(
                qa_rows=qa_rows,
                existing_flag_ids=existing_flag_ids,
                table_id=table_id,
                page=page,
                flag_type="llm_rectification_invalid_schema",
                severity="warn",
                details=schema_reason,
            )
            result.table_results.append(
                {
                    "table_id": table_id,
                    "page": page,
                    "risk_score": risk_score,
                    "status": "invalid_schema",
                    "reason": schema_reason,
                    "context_requested": context_requested,
                    "context_window": context_window_used,
                }
            )
            continue

        ok_evidence, evidence_reason, normalized_provenance, repaired_headers, repaired_rows = _validate_evidence(
            table_payload=table_payload,
            normalized=normalized,
            evidence=evidence,
            table_structure=table_structure,
            structure_lock=bool(config.structure_lock),
        )
        if not ok_evidence and evidence_reason.startswith("evidence_violation:"):
            retry_raw: dict[str, Any] = dict(parsed)
            retry_prompt = _build_evidence_retry_prompt(
                evidence_reason=evidence_reason,
                previous_output=retry_raw,
                table_payload=table_payload,
                manifest_row=manifest_row,
                evidence=evidence,
                validation_feedback=validation_feedback,
                table_structure=table_structure,
            )
            try:
                retry_response = await call_model(
                    client=client,
                    model=config.model,
                    prompt=retry_prompt,
                    max_tokens=int(config.max_tokens),
                )
                retry_parsed = _extract_json_object(getattr(retry_response, "content", ""))
                if not retry_parsed:
                    repair = await call_model(
                        client=client,
                        model=config.model,
                        prompt=_build_repair_prompt(getattr(retry_response, "content", "")),
                        max_tokens=int(config.max_tokens),
                    )
                    retry_parsed = _extract_json_object(getattr(repair, "content", ""))
                if retry_parsed:
                    ok_schema_retry, schema_reason_retry, normalized_retry = _normalize_rectified_payload(retry_parsed)
                    if ok_schema_retry:
                        ok_evidence, evidence_reason, normalized_provenance, repaired_headers, repaired_rows = _validate_evidence(
                            table_payload=table_payload,
                            normalized=normalized_retry,
                            evidence=evidence,
                            table_structure=table_structure,
                            structure_lock=bool(config.structure_lock),
                        )
                        if ok_evidence:
                            normalized = normalized_retry
            except Exception:
                pass

        if not ok_evidence:
            result.fallbacked += 1
            if evidence_reason.startswith("evidence_violation:"):
                result.evidence_violations += 1
                parse_status = "evidence_violation"
                _append_flag(
                    qa_rows=qa_rows,
                    existing_flag_ids=existing_flag_ids,
                    table_id=table_id,
                    page=page,
                    flag_type="llm_rectification_evidence_violation",
                    severity="warn",
                    details=evidence_reason,
                )
            else:
                result.invalid_schema += 1
                parse_status = "invalid_schema"
                _append_flag(
                    qa_rows=qa_rows,
                    existing_flag_ids=existing_flag_ids,
                    table_id=table_id,
                    page=page,
                    flag_type="llm_rectification_invalid_schema",
                    severity="warn",
                    details=evidence_reason,
                )
            result.table_results.append(
                {
                    "table_id": table_id,
                    "page": page,
                    "risk_score": risk_score,
                    "status": parse_status,
                    "reason": evidence_reason,
                    "structure_used": bool(table_structure),
                    "context_requested": context_requested,
                    "context_window": context_window_used,
                    "context_request_reason": context_request_reason,
                }
            )
            continue

        normalized["headers"] = repaired_headers
        normalized["rows"] = repaired_rows
        normalized["header_rows_full"] = _resolve_header_rows_full_for_apply(
            candidate_header_rows_full=normalized.get("header_rows_full", []),
            fallback_header_rows_full=table_payload.get("header_rows_full", []),
            headers=repaired_headers,
        )
        normalized["cell_provenance"] = normalized_provenance
        updated_payload, headers, rows = _apply_rectification(
            doc_dir=doc_dir,
            table_payload=table_payload,
            normalized=normalized,
            risk_score=risk_score,
            config=config,
        )
        table_json_path = tables_dir / f"{table_id}.json"
        table_json_path.write_text(json.dumps(updated_payload, indent=2, ensure_ascii=True))

        if table_id in manifest_by_id:
            row = dict(manifest_by_id[table_id])
            row["headers"] = headers
            row["rows"] = rows
            manifest_by_id[table_id] = row
        if table_id in canonical_by_id:
            crow = dict(canonical_by_id[table_id])
            crow["header_rows"] = [headers]
            crow["header_rows_full"] = updated_payload.get("header_rows_full", [headers])
            crow["header_hierarchy"] = updated_payload.get("header_hierarchy", [])
            crow["data_rows"] = rows
            crow["quality_metrics"] = updated_payload.get("quality_metrics", {})
            crow["cell_provenance"] = updated_payload.get("cell_provenance", [])
            crow["rectifier_confidence"] = updated_payload.get("rectifier_confidence", 0.0)
            crow["rectifier_needs_review"] = updated_payload.get("rectifier_needs_review", True)
            crow["llm_rectification"] = updated_payload.get("llm_rectification", {})
            canonical_by_id[table_id] = crow

        result.applied += 1
        if used_schema_noop_fallback:
            _append_flag(
                qa_rows=qa_rows,
                existing_flag_ids=existing_flag_ids,
                table_id=table_id,
                page=page,
                flag_type="llm_rectification_schema_fallback_noop",
                severity="warn",
                details="invalid_schema_empty_or_missing_grid",
            )
        _append_flag(
            qa_rows=qa_rows,
            existing_flag_ids=existing_flag_ids,
            table_id=table_id,
            page=page,
            flag_type="llm_rectification_applied",
            severity="info",
            details=f"risk={risk_score:.4f}",
        )
        result.table_results.append(
            {
                "table_id": table_id,
                "page": page,
                "risk_score": risk_score,
                "status": "applied",
                "structure_used": bool(table_structure),
                "context_requested": context_requested,
                "context_window": context_window_used,
                "context_request_reason": context_request_reason,
                "rectifier_confidence": float(updated_payload.get("rectifier_confidence", 0.0) or 0.0),
                "needs_review": bool(updated_payload.get("rectifier_needs_review", False)),
            }
        )

    # Persist updates after processing all candidates.
    updated_manifest_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        table_id = str(row.get("table_id", ""))
        updated_manifest_rows.append(dict(manifest_by_id.get(table_id, row)))
    _write_jsonl(tables_manifest_path, updated_manifest_rows)

    if canonical_rows:
        updated_canonical_rows: list[dict[str, Any]] = []
        for row in canonical_rows:
            table_id = str(row.get("table_id", ""))
            updated_canonical_rows.append(dict(canonical_by_id.get(table_id, row)))
        _write_jsonl(canonical_path, updated_canonical_rows)

    # Persist run-scoped canonical snapshots for benchmarking across reruns.
    input_snapshot_rows: list[dict[str, Any]] = []
    output_snapshot_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        table_id = str(row.get("table_id", "")).strip()
        if not table_id:
            continue
        baseline_payload = dict(table_payloads.get(table_id, {}))
        if baseline_payload:
            input_snapshot_rows.append(
                {
                    "table_id": table_id,
                    "page": int(row.get("page", 0) or 0),
                    "caption_text": str(baseline_payload.get("caption_text", row.get("caption", ""))),
                    "header_rows_full": baseline_payload.get("header_rows_full", []),
                    "data_rows": baseline_payload.get("data_rows", []),
                    "quality_metrics": baseline_payload.get("quality_metrics", {}),
                }
            )
        current_payload = _load_json(tables_dir / f"{table_id}.json")
        if current_payload:
            output_snapshot_rows.append(
                {
                    "table_id": table_id,
                    "page": int(row.get("page", 0) or 0),
                    "caption_text": str(current_payload.get("caption_text", row.get("caption", ""))),
                    "header_rows_full": current_payload.get("header_rows_full", []),
                    "data_rows": current_payload.get("data_rows", []),
                    "quality_metrics": current_payload.get("quality_metrics", {}),
                    "llm_rectification": current_payload.get("llm_rectification", {}),
                }
            )
    _write_jsonl(tables_dir / f"canonical.input.{run_id}.jsonl", input_snapshot_rows)
    _write_jsonl(tables_dir / f"canonical.rectified.{run_id}.jsonl", output_snapshot_rows)

    _write_jsonl(qa_flags_path, qa_rows)
    report_payload = {
        "enabled": result.enabled,
        "model": result.model,
        "target": result.target,
        "run_id": result.run_id,
        "considered": result.considered,
        "selected": result.selected,
        "applied": result.applied,
        "skipped_low_risk": result.skipped_low_risk,
        "fallbacked": result.fallbacked,
        "invalid_schema": result.invalid_schema,
        "evidence_violations": result.evidence_violations,
        "errors": result.errors,
        "report_path": result.report_path,
        "table_results": result.table_results,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True))
    return result
