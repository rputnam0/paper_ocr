from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from google import genai
from google.genai import types


TABLE_VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "table_present_on_page": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "extraction_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "false_positive_risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "recommended_action": {"type": "string", "enum": ["accept", "review", "reject"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "failure_modes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "split_table_continuation",
                    "header_misalignment",
                    "row_shift_or_merge",
                    "column_shift_or_merge",
                    "multi_level_header_loss",
                    "symbol_or_unit_loss",
                    "caption_mismatch",
                    "code_legend_unresolved",
                    "partial_table_extracted",
                    "duplicate_or_hallucinated_rows",
                    "missing_required_columns",
                    "table_not_present",
                    "ocr_noise",
                    "other",
                ],
            },
        },
        "root_cause_hypothesis": {"type": "string"},
        "needs_followup": {"type": "boolean"},
        "followup_recommendations": {"type": "array", "items": {"type": "string"}},
        "llm_extraction_instructions": {"type": "array", "items": {"type": "string"}},
        "missing_required_information": {"type": "array", "items": {"type": "string"}},
        "formatting_issues": {"type": "array", "items": {"type": "string"}},
        "rubric": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "formatting_fidelity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "structural_fidelity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "data_completeness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "unit_symbol_fidelity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "context_resolution": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "overall_robustness": {
                    "type": "string",
                    "enum": ["robust", "mostly_robust", "fragile", "failed"],
                },
            },
            "required": [
                "formatting_fidelity",
                "structural_fidelity",
                "data_completeness",
                "unit_symbol_fidelity",
                "context_resolution",
                "overall_robustness",
            ],
        },
    },
    "required": [
        "table_present_on_page",
        "extraction_quality",
        "false_positive_risk",
        "recommended_action",
        "issues",
        "failure_modes",
        "root_cause_hypothesis",
        "needs_followup",
        "followup_recommendations",
        "llm_extraction_instructions",
        "missing_required_information",
        "formatting_issues",
        "rubric",
    ],
}


@dataclass
class GeminiValidationConfig:
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    max_output_tokens: int = 2000
    render_dpi: int = 220
    thinking_budget: int = 0


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


def _discover_ocr_doc_dirs(ocr_out_dir: Path) -> list[Path]:
    if (ocr_out_dir / "metadata").is_dir() and ocr_out_dir.name.startswith("doc_"):
        return [ocr_out_dir]

    out: list[Path] = []
    seen: set[str] = set()
    for manifest_path in sorted(ocr_out_dir.rglob("metadata/manifest.json")):
        doc_dir = manifest_path.parent.parent
        if not doc_dir.name.startswith("doc_"):
            continue
        key = str(doc_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(doc_dir)
    return out


def _is_problem_doc(doc_dir: Path) -> bool:
    status_path = doc_dir / "metadata" / "assets" / "structured" / "qa" / "pipeline_status.json"
    payload = _load_json(status_path)
    status = str(payload.get("status", "")).strip().lower()
    if not status:
        return True
    return status != "ok"


def _resolve_source_pdf(manifest: dict[str, Any], doc_dir: Path) -> Path | None:
    source_raw = str(manifest.get("source_path", "")).strip()
    if not source_raw:
        return None
    source_path = Path(source_raw)
    candidates = [source_path]
    if not source_path.is_absolute():
        candidates.extend([Path.cwd() / source_path, doc_dir / source_path, doc_dir.parent / source_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _canonical_model_id(model: str) -> str:
    requested = str(model or "").strip()
    if not requested:
        return "gemini-2.5-flash"
    if requested.startswith("models/"):
        requested = requested.split("/", 1)[1]
    aliases = {
        "gemini-3-flash": "gemini-3-flash-preview",
    }
    return aliases.get(requested, requested)


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
        return _extract_partial_review_fields(raw)
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:
        return _extract_partial_review_fields(raw)
    return parsed if isinstance(parsed, dict) else _extract_partial_review_fields(raw)


def _extract_partial_review_fields(text: str) -> dict[str, Any]:
    raw = str(text or "")
    if not raw:
        return {}
    out: dict[str, Any] = {}
    m_presence = re.search(
        r'"table_present_on_page"\s*:\s*(?:"(yes|no|uncertain|true|false)"|(true|false))',
        raw,
        flags=re.IGNORECASE,
    )
    if m_presence:
        out["table_present_on_page"] = (m_presence.group(1) or m_presence.group(2) or "").lower()
    m_quality = re.search(r'"extraction_quality"\s*:\s*([0-9]*\.?[0-9]+)', raw, flags=re.IGNORECASE)
    if m_quality:
        out["extraction_quality"] = float(m_quality.group(1))
    m_risk = re.search(r'"false_positive_risk"\s*:\s*([0-9]*\.?[0-9]+)', raw, flags=re.IGNORECASE)
    if m_risk:
        out["false_positive_risk"] = float(m_risk.group(1))
    m_action = re.search(r'"recommended_action"\s*:\s*"(accept|review|reject)"', raw, flags=re.IGNORECASE)
    if m_action:
        out["recommended_action"] = m_action.group(1).lower()
    m_issues = re.search(r'"issues"\s*:\s*\[(.*?)\]', raw, flags=re.IGNORECASE | re.DOTALL)
    if m_issues:
        issues: list[str] = []
        for m in re.finditer(r'"([^"]+)"', m_issues.group(1)):
            issue = m.group(1).strip()
            if issue:
                issues.append(issue)
        out["issues"] = issues
    m_failure = re.search(r'"failure_modes"\s*:\s*\[(.*?)\]', raw, flags=re.IGNORECASE | re.DOTALL)
    if m_failure:
        modes: list[str] = []
        for m in re.finditer(r'"([^"]+)"', m_failure.group(1)):
            mode = m.group(1).strip()
            if mode:
                modes.append(mode)
        out["failure_modes"] = modes
    m_missing = re.search(r'"missing_required_information"\s*:\s*\[(.*?)\]', raw, flags=re.IGNORECASE | re.DOTALL)
    if m_missing:
        missing: list[str] = []
        for m in re.finditer(r'"([^"]+)"', m_missing.group(1)):
            item = m.group(1).strip()
            if item:
                missing.append(item)
        out["missing_required_information"] = missing
    m_llm = re.search(r'"llm_extraction_instructions"\s*:\s*\[(.*?)\]', raw, flags=re.IGNORECASE | re.DOTALL)
    if m_llm:
        llm: list[str] = []
        for m in re.finditer(r'"([^"]+)"', m_llm.group(1)):
            item = m.group(1).strip()
            if item:
                llm.append(item)
        out["llm_extraction_instructions"] = llm
    return out


def _clamp01(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _normalize_review(payload: dict[str, Any], *, raw_response: str) -> dict[str, Any]:
    raw_presence = payload.get("table_present_on_page", "uncertain")
    if isinstance(raw_presence, bool):
        presence = "yes" if raw_presence else "no"
    else:
        presence = str(raw_presence).strip().lower()
    if presence in {"true", "present"}:
        presence = "yes"
    if presence in {"false", "absent"}:
        presence = "no"
    if presence not in {"yes", "no", "uncertain"}:
        presence = "uncertain"

    action = str(payload.get("recommended_action", "review")).strip().lower()
    if action not in {"accept", "review", "reject"}:
        action = "review"

    quality = _clamp01(payload.get("extraction_quality", 0.0), default=0.0)
    risk_raw = payload.get("false_positive_risk", None)
    if risk_raw is None:
        risk = _clamp01(1.0 - quality, default=1.0)
    else:
        risk = _clamp01(risk_raw, default=1.0)

    issues_raw = payload.get("issues", [])
    issues: list[str] = []
    if isinstance(issues_raw, list):
        for item in issues_raw:
            text = str(item).strip()
            if text:
                issues.append(text)

    failure_modes_raw = payload.get("failure_modes", [])
    failure_modes: list[str] = []
    if isinstance(failure_modes_raw, list):
        for item in failure_modes_raw:
            text = str(item).strip()
            if text:
                failure_modes.append(text)

    followup_raw = payload.get("followup_recommendations", [])
    followup_recommendations: list[str] = []
    if isinstance(followup_raw, list):
        for item in followup_raw:
            text = str(item).strip()
            if text:
                followup_recommendations.append(text)

    llm_instr_raw = payload.get("llm_extraction_instructions", [])
    llm_extraction_instructions: list[str] = []
    if isinstance(llm_instr_raw, list):
        for item in llm_instr_raw:
            text = str(item).strip()
            if text:
                llm_extraction_instructions.append(text)

    missing_info_raw = payload.get("missing_required_information", [])
    missing_required_information: list[str] = []
    if isinstance(missing_info_raw, list):
        for item in missing_info_raw:
            text = str(item).strip()
            if text:
                missing_required_information.append(text)

    formatting_issues_raw = payload.get("formatting_issues", [])
    formatting_issues: list[str] = []
    if isinstance(formatting_issues_raw, list):
        for item in formatting_issues_raw:
            text = str(item).strip()
            if text:
                formatting_issues.append(text)

    rubric_payload = payload.get("rubric", {})
    if not isinstance(rubric_payload, dict):
        rubric_payload = {}
    rubric = {
        "formatting_fidelity": _clamp01(rubric_payload.get("formatting_fidelity", quality), default=quality),
        "structural_fidelity": _clamp01(rubric_payload.get("structural_fidelity", quality), default=quality),
        "data_completeness": _clamp01(rubric_payload.get("data_completeness", quality), default=quality),
        "unit_symbol_fidelity": _clamp01(rubric_payload.get("unit_symbol_fidelity", quality), default=quality),
        "context_resolution": _clamp01(rubric_payload.get("context_resolution", quality), default=quality),
        "overall_robustness": str(rubric_payload.get("overall_robustness", "") or "").strip().lower(),
    }
    if rubric["overall_robustness"] not in {"robust", "mostly_robust", "fragile", "failed"}:
        if quality >= 0.9:
            rubric["overall_robustness"] = "robust"
        elif quality >= 0.75:
            rubric["overall_robustness"] = "mostly_robust"
        elif quality >= 0.45:
            rubric["overall_robustness"] = "fragile"
        else:
            rubric["overall_robustness"] = "failed"

    return {
        "table_present_on_page": presence,
        "extraction_quality": quality,
        "false_positive_risk": risk,
        "recommended_action": action,
        "issues": issues,
        "failure_modes": failure_modes,
        "root_cause_hypothesis": str(payload.get("root_cause_hypothesis", "") or "").strip(),
        "needs_followup": bool(payload.get("needs_followup", False)),
        "followup_recommendations": followup_recommendations,
        "llm_extraction_instructions": llm_extraction_instructions,
        "missing_required_information": missing_required_information,
        "formatting_issues": formatting_issues,
        "rubric": rubric,
        "raw_response": raw_response,
    }


def _short(value: Any, *, limit: int = 80) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _resolve_csv_text(doc_dir: Path, table_row: dict[str, Any]) -> str:
    csv_path_raw = str(table_row.get("csv_path", "")).strip()
    if not csv_path_raw:
        return ""
    csv_path = Path(csv_path_raw)
    candidates = [csv_path]
    if not csv_path.is_absolute():
        candidates.append(doc_dir / csv_path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                return candidate.read_text()
            except Exception:
                return ""
    return ""


def _resolve_fragment_index(doc_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    fragments_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "table_fragments.jsonl"
    for row in _load_jsonl(fragments_path):
        fid = str(row.get("fragment_id", "")).strip()
        if fid:
            out[fid] = row
    return out


def _resolve_table_validation_context(
    *,
    doc_dir: Path,
    table_row: dict[str, Any],
    fragment_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    table_id = str(table_row.get("table_id", "")).strip()
    detail_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / f"{table_id}.json"
    detail = _load_json(detail_path) if table_id else {}
    csv_text = _resolve_csv_text(doc_dir, table_row)

    fragment_rows: list[dict[str, Any]] = []
    fragment_ids = detail.get("fragment_ids", [])
    if isinstance(fragment_ids, list):
        for item in fragment_ids:
            fid = str(item).strip()
            frag = fragment_index.get(fid)
            if not frag:
                continue
            fragment_rows.append(
                {
                    "fragment_id": fid,
                    "page": frag.get("page"),
                    "table_group_id": frag.get("table_group_id"),
                    "header_rows": frag.get("header_rows", []),
                    "data_rows": frag.get("data_rows", []),
                    "caption_text": frag.get("caption_text", ""),
                }
            )

    ocr_html = ""
    ocr_merge = detail.get("ocr_merge", {}) if isinstance(detail, dict) else {}
    if isinstance(ocr_merge, dict):
        ocr_path_raw = str(ocr_merge.get("ocr_html_path", "")).strip()
        if ocr_path_raw:
            ocr_path = Path(ocr_path_raw)
            candidates = [ocr_path]
            if not ocr_path.is_absolute():
                candidates.append(doc_dir / ocr_path)
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    try:
                        ocr_html = candidate.read_text()
                    except Exception:
                        ocr_html = ""
                    break

    return {
        "table_detail": detail,
        "extracted_csv": csv_text,
        "marker_fragments": fragment_rows,
        "ocr_html": ocr_html,
    }


def _resolve_validation_pages(table_row: dict[str, Any], doc_dir: Path) -> list[int]:
    primary = int(table_row.get("page", 0) or 0)
    pages: list[int] = [primary] if primary > 0 else []
    table_id = str(table_row.get("table_id", "")).strip()
    if not table_id:
        return pages
    table_detail_path = (
        doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / f"{table_id}.json"
    )
    detail = _load_json(table_detail_path)
    raw_pages = detail.get("pages", [])
    if isinstance(raw_pages, list):
        parsed: list[int] = []
        for item in raw_pages:
            try:
                page = int(item)
            except Exception:
                continue
            if page > 0:
                parsed.append(page)
        if parsed:
            pages = parsed
    deduped: list[int] = []
    seen: set[int] = set()
    for page in pages:
        if page in seen:
            continue
        seen.add(page)
        deduped.append(page)
    return deduped


GROUP_PAGE_TABLE_RE = re.compile(r"page_(\d+)_table_(\d+)$")


def _resolve_table_crop_paths(
    *,
    doc_dir: Path,
    validation_pages: list[int],
    context: dict[str, Any],
) -> list[Path]:
    crops_root = doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_crops"
    if not crops_root.exists():
        return []
    paths: list[Path] = []
    seen: set[str] = set()

    fragments = context.get("marker_fragments", [])
    if isinstance(fragments, list):
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            page = int(frag.get("page", 0) or 0)
            group = str(frag.get("table_group_id", "")).strip()
            m = GROUP_PAGE_TABLE_RE.search(group)
            if page <= 0 or not m:
                continue
            ordinal = int(m.group(2))
            crop_path = crops_root / f"table_{ordinal:02d}_page_{page:04d}.png"
            key = str(crop_path)
            if crop_path.exists() and key not in seen:
                seen.add(key)
                paths.append(crop_path)

    if paths:
        return paths

    # Fallback: include all crops on continuation pages when fragment mapping is absent.
    for page in validation_pages:
        for path in sorted(crops_root.glob(f"table_*_page_{int(page):04d}.png")):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _build_review_prompt(table_row: dict[str, Any]) -> str:
    raw_headers = table_row.get("headers", [])
    headers = [str(x) for x in raw_headers] if isinstance(raw_headers, list) else []
    raw_rows = table_row.get("rows", [])
    rows: list[list[str]] = []
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if isinstance(row, list):
                rows.append([str(cell) for cell in row])

    extracted_table_json = _truncate_text(
        json.dumps({"headers": headers, "rows": rows}, ensure_ascii=True),
        limit=32000,
    )
    extracted_csv_text = _truncate_text(str(table_row.get("extracted_csv", "") or ""), limit=32000)
    marker_fragments_json = _truncate_text(
        json.dumps(table_row.get("marker_fragments", []), ensure_ascii=True),
        limit=24000,
    )
    ocr_html_text = _truncate_text(str(table_row.get("ocr_html", "") or ""), limit=12000)
    table_detail_json = _truncate_text(
        json.dumps(table_row.get("table_detail", {}), ensure_ascii=True),
        limit=16000,
    )

    return (
        "Task: Validate one extracted table candidate against the provided page image(s).\n"
        "Important: If multiple page images are provided, they represent continuation pages of the same table.\n"
        "Use all provided evidence (page images, optional table crop images, marker fragments, OCR html, and final extracted table).\n"
        "Judge whether final extraction preserves structure, formatting, headers, rows, symbols/units, and continuation semantics.\n"
        "If table uses encoded labels/codes not resolved in the extracted output, flag code_legend_unresolved.\n"
        "Also determine if required information is missing (units, legend/code mappings, key columns, continuation rows, or references required to interpret rows).\n"
        "Return JSON only (no markdown) with keys exactly:\n"
        'table_present_on_page ("yes"|"no"|"uncertain"),\n'
        "extraction_quality (0..1),\n"
        "false_positive_risk (0..1),\n"
        'recommended_action ("accept"|"review"|"reject"),\n'
        "issues (string array),\n"
        "failure_modes (string array),\n"
        "root_cause_hypothesis (string),\n"
        "needs_followup (boolean),\n"
        "followup_recommendations (string array),\n"
        "llm_extraction_instructions (string array of concrete prompt instructions for a future LLM-based extractor),\n"
        "missing_required_information (string array),\n"
        "formatting_issues (string array),\n"
        'rubric (object with formatting_fidelity, structural_fidelity, data_completeness, unit_symbol_fidelity, context_resolution in [0..1], and overall_robustness in {"robust","mostly_robust","fragile","failed"}).\n'
        "Scoring guidance:\n"
        "- Accept only when table structure/cells are mostly correct.\n"
        "- Review when table exists but extraction has notable issues.\n"
        "- Reject when table candidate is mostly wrong.\n"
        "Rubric guidance:\n"
        "- formatting_fidelity: CSV/layout faithfulness to table formatting intent.\n"
        "- structural_fidelity: header/column/row alignment and hierarchy correctness.\n"
        "- data_completeness: whether rows/columns/cells are missing or hallucinated.\n"
        "- unit_symbol_fidelity: units/symbols/superscripts/subscripts/signs preserved.\n"
        "- context_resolution: whether aliases/codes/legend references are resolved enough to interpret data.\n"
        "For llm_extraction_instructions:\n"
        "- Provide 2-6 concise, imperative instructions that could be inserted into an extraction prompt.\n"
        "- Focus on fixing the observed failure mechanisms for this table.\n"
        "- Mention exact needs (e.g., preserve multi-row headers, resolve polymer alias codes from legend text, keep units/symbols verbatim).\n"
        "Failure mode taxonomy:\n"
        "- split_table_continuation, header_misalignment, row_shift_or_merge, column_shift_or_merge,\n"
        "  multi_level_header_loss, symbol_or_unit_loss, caption_mismatch, code_legend_unresolved,\n"
        "  partial_table_extracted, duplicate_or_hallucinated_rows, missing_required_columns,\n"
        "  table_not_present, ocr_noise, other.\n"
        "Candidate summary:\n"
        f"table_id={_short(table_row.get('table_id', ''), limit=32)}\n"
        f"page={int(table_row.get('page', 0) or 0)}\n"
        f"validation_pages={table_row.get('validation_pages', [])}\n"
        f"caption={_short(table_row.get('caption', ''), limit=180)}\n"
        "Final extracted table (JSON, full where possible):\n"
        f"{extracted_table_json}\n"
        "Final extracted CSV text:\n"
        f"{extracted_csv_text}\n"
        "Canonical table detail JSON:\n"
        f"{table_detail_json}\n"
        "Marker fragment lineage (JSON):\n"
        f"{marker_fragments_json}\n"
        "OCR merged table html/text (if available):\n"
        f"{ocr_html_text}"
    )


def _generate_config(config: GeminiValidationConfig) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0,
        maxOutputTokens=int(config.max_output_tokens),
        responseMimeType="application/json",
        responseJsonSchema=TABLE_VALIDATION_SCHEMA,
        thinkingConfig=types.ThinkingConfig(thinkingBudget=int(config.thinking_budget)),
    )


async def _review_table_with_gemini(
    *,
    client: Any,
    config: GeminiValidationConfig,
    page_image_bytes_list: list[bytes],
    crop_image_bytes_list: list[bytes],
    page_numbers: list[int],
    table_row: dict[str, Any],
) -> dict[str, Any]:
    model = _canonical_model_id(config.model)
    prompt = _build_review_prompt(table_row)

    # Test clients can provide a lightweight async method with the same semantic contract.
    if hasattr(client, "generate_table_review"):
        raw_response = await client.generate_table_review(
            model=model,
            prompt=prompt,
            image_bytes=page_image_bytes_list[0] if page_image_bytes_list else b"",
            image_bytes_list=page_image_bytes_list + crop_image_bytes_list,
            page_image_bytes_list=page_image_bytes_list,
            crop_image_bytes_list=crop_image_bytes_list,
            page_numbers=page_numbers,
            generation_config=_generate_config(config),
        )
        parsed = _extract_json_object(str(raw_response))
        review = _normalize_review(parsed, raw_response=str(raw_response))
        review["model_used"] = model
        return review

    content_parts: list[types.Part] = [types.Part.from_text(text=prompt)]
    for idx, image_bytes in enumerate(page_image_bytes_list, start=1):
        content_parts.append(types.Part.from_text(text=f"Page image {idx}/{len(page_image_bytes_list)}"))
        content_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    for idx, image_bytes in enumerate(crop_image_bytes_list, start=1):
        content_parts.append(types.Part.from_text(text=f"Table crop image {idx}/{len(crop_image_bytes_list)}"))
        content_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

    response = await client.aio.models.generate_content(
        model=model,
        contents=content_parts,
        config=_generate_config(config),
    )
    content = str(getattr(response, "text", "") or "")
    parsed = _extract_json_object(content)
    review = _normalize_review(parsed, raw_response=content)
    review["model_used"] = model
    try:
        raw = response.model_dump()
        candidate = (raw.get("candidates") or [{}])[0]
        review["finish_reason"] = str(candidate.get("finish_reason", "") or "")
    except Exception:
        review["finish_reason"] = ""
    return review


def _render_page_png(pdf: fitz.Document, page_number: int, dpi: int) -> bytes:
    page_idx = int(page_number) - 1
    if page_idx < 0 or page_idx >= int(pdf.page_count):
        raise ValueError(f"invalid_page:{page_number}")
    page = pdf.load_page(page_idx)
    scale = max(float(dpi), 72.0) / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _infer_metrics_report_path(ocr_out_dir: Path) -> Path:
    base_name = ocr_out_dir.name
    if base_name.lower() == "pdfs" and ocr_out_dir.parent.name:
        base_name = ocr_out_dir.parent.name
    clean_slug = re.sub(r"_\d{8}_\d{6}$", "", base_name)
    candidates = [
        Path("data/jobs") / clean_slug / "reports",
        Path("data/jobs") / ocr_out_dir.name / "reports",
    ]
    report_root = next((path for path in candidates if path.exists()), candidates[0])
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return report_root / f"table_fix_backlog_metrics_{stamp}.json"


def _extract_action_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("action_counts", {})
    if isinstance(counts, dict) and counts:
        return {
            "accept": int(counts.get("accept", 0) or 0),
            "review": int(counts.get("review", 0) or 0),
            "reject": int(counts.get("reject", 0) or 0),
        }
    legacy = payload.get("actions", {})
    if isinstance(legacy, dict):
        return {
            "accept": int(legacy.get("accept", 0) or 0),
            "review": int(legacy.get("review", 0) or 0),
            "reject": int(legacy.get("reject", 0) or 0),
        }
    return {"accept": 0, "review": 0, "reject": 0}


def _extract_failure_mode_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("failure_mode_counts", {})
    if isinstance(counts, dict) and counts:
        return {str(k): int(v or 0) for k, v in counts.items()}
    top = payload.get("top_failure_modes", [])
    if isinstance(top, list):
        out: dict[str, int] = {}
        for row in top:
            if isinstance(row, list) and len(row) >= 2:
                key = str(row[0]).strip()
                if key:
                    out[key] = int(row[1] or 0)
                continue
            if not isinstance(row, dict):
                continue
            key = str(row.get("mode", "")).strip()
            if not key:
                continue
            out[key] = int(row.get("count", 0) or 0)
        return out
    return {}


def _extract_rubric_averages(payload: dict[str, Any]) -> dict[str, float]:
    rubric = payload.get("rubric_averages", {})
    if isinstance(rubric, dict) and rubric:
        return {str(k): _safe_float(v, default=0.0) for k, v in rubric.items()}
    return {}


def _extract_robustness_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = payload.get("robustness_counts", {})
    if isinstance(counts, dict):
        return {str(k): int(v or 0) for k, v in counts.items()}
    return {}


def _evaluate_backlog_gates(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    gates: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "passed": True, "checks": []}
    if not baseline or not gates:
        return out
    out["available"] = True
    current_actions = _extract_action_counts(current)
    baseline_actions = _extract_action_counts(baseline)
    current_modes = _extract_failure_mode_counts(current)
    baseline_modes = _extract_failure_mode_counts(baseline)
    current_rubric = _extract_rubric_averages(current)
    baseline_rubric = _extract_rubric_averages(baseline)

    global_targets = gates.get("global_targets", {}) if isinstance(gates.get("global_targets", {}), dict) else {}
    reject_max = int(global_targets.get("recommended_reject_max", 10**9))
    accept_delta_min = int(global_targets.get("recommended_accept_delta_min", -10**9))
    table_present_no_max = int(global_targets.get("table_present_no_max", 10**9))
    current_table_present_no = int(current.get("table_present_no", 0) or 0)

    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "recommended_reject_max",
            "actual": int(current_actions.get("reject", 0)),
            "target": reject_max,
            "pass": int(current_actions.get("reject", 0)) <= reject_max,
        }
    )
    checks.append(
        {
            "name": "recommended_accept_delta_min",
            "actual": int(current_actions.get("accept", 0)) - int(baseline_actions.get("accept", 0)),
            "target": accept_delta_min,
            "pass": (int(current_actions.get("accept", 0)) - int(baseline_actions.get("accept", 0))) >= accept_delta_min,
        }
    )
    checks.append(
        {
            "name": "table_present_no_max",
            "actual": current_table_present_no,
            "target": table_present_no_max,
            "pass": current_table_present_no <= table_present_no_max,
        }
    )

    mode_targets = gates.get("failure_mode_reduction_targets", {})
    if isinstance(mode_targets, dict):
        for mode, reduction_min in mode_targets.items():
            base_count = int(baseline_modes.get(str(mode), 0))
            current_count = int(current_modes.get(str(mode), 0))
            if base_count <= 0:
                passed = True
                actual_reduction = 1.0 if current_count == 0 else 0.0
            else:
                actual_reduction = max(0.0, (base_count - current_count) / float(base_count))
                passed = actual_reduction >= _safe_float(reduction_min, default=0.0)
            checks.append(
                {
                    "name": f"failure_mode_reduction:{mode}",
                    "actual": round(actual_reduction, 3),
                    "target": _safe_float(reduction_min, default=0.0),
                    "baseline_count": base_count,
                    "current_count": current_count,
                    "pass": passed,
                }
            )

    rubric_targets = gates.get("rubric_targets", {})
    if isinstance(rubric_targets, dict):
        if "data_completeness_delta_min" in rubric_targets:
            delta = _safe_float(current_rubric.get("data_completeness", 0.0)) - _safe_float(
                baseline_rubric.get("data_completeness", 0.0)
            )
            target = _safe_float(rubric_targets.get("data_completeness_delta_min", 0.0))
            checks.append({"name": "rubric:data_completeness_delta_min", "actual": round(delta, 3), "target": target, "pass": delta >= target})
        if "unit_symbol_fidelity_min" in rubric_targets:
            actual = _safe_float(current_rubric.get("unit_symbol_fidelity", 0.0))
            target = _safe_float(rubric_targets.get("unit_symbol_fidelity_min", 0.0))
            checks.append({"name": "rubric:unit_symbol_fidelity_min", "actual": round(actual, 3), "target": target, "pass": actual >= target})
        if "context_resolution_delta_min" in rubric_targets:
            delta = _safe_float(current_rubric.get("context_resolution", 0.0)) - _safe_float(
                baseline_rubric.get("context_resolution", 0.0)
            )
            target = _safe_float(rubric_targets.get("context_resolution_delta_min", 0.0))
            checks.append({"name": "rubric:context_resolution_delta_min", "actual": round(delta, 3), "target": target, "pass": delta >= target})

    out["checks"] = checks
    out["passed"] = all(bool(row.get("pass", False)) for row in checks)
    return out


def summarize_gemini_failures(
    *,
    ocr_out_dir: Path,
    report_out: Path | None = None,
    baseline_path: Path | None = None,
    gates_path: Path | None = None,
) -> dict[str, Any]:
    docs = _discover_ocr_doc_dirs(ocr_out_dir)
    action_counts = {"accept": 0, "review": 0, "reject": 0}
    failure_mode_counts: dict[str, int] = {}
    robustness_counts = {"robust": 0, "mostly_robust": 0, "fragile": 0, "failed": 0}
    table_present_yes = 0
    table_present_no = 0
    table_present_uncertain = 0
    rubric_sums = {
        "formatting_fidelity": 0.0,
        "structural_fidelity": 0.0,
        "data_completeness": 0.0,
        "unit_symbol_fidelity": 0.0,
        "context_resolution": 0.0,
    }
    tables_reviewed = 0

    for doc_dir in docs:
        review_path = doc_dir / "metadata" / "assets" / "structured" / "validation" / "gemini_table_review.jsonl"
        rows = _load_jsonl(review_path)
        for row in rows:
            review = row.get("model_review", {})
            if not isinstance(review, dict):
                continue
            tables_reviewed += 1
            action = str(review.get("recommended_action", "review")).strip().lower()
            if action not in action_counts:
                action = "review"
            action_counts[action] += 1
            presence = str(review.get("table_present_on_page", "uncertain")).strip().lower()
            if presence == "yes":
                table_present_yes += 1
            elif presence == "no":
                table_present_no += 1
            else:
                table_present_uncertain += 1
            for mode in review.get("failure_modes", []):
                key = str(mode).strip()
                if not key:
                    continue
                failure_mode_counts[key] = int(failure_mode_counts.get(key, 0)) + 1
            rubric = review.get("rubric", {})
            if isinstance(rubric, dict):
                robustness = str(rubric.get("overall_robustness", "")).strip().lower()
                if robustness in robustness_counts:
                    robustness_counts[robustness] += 1
                for key in rubric_sums:
                    rubric_sums[key] += _safe_float(rubric.get(key, 0.0), default=0.0)

    rubric_averages = {key: 0.0 for key in rubric_sums}
    if tables_reviewed > 0:
        rubric_averages = {key: round(value / float(tables_reviewed), 3) for key, value in rubric_sums.items()}

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ocr_out_dir": str(ocr_out_dir),
        "docs_discovered": len(docs),
        "tables_reviewed": tables_reviewed,
        "table_present_yes": table_present_yes,
        "table_present_no": table_present_no,
        "table_present_uncertain": table_present_uncertain,
        "action_counts": action_counts,
        "failure_mode_counts": dict(sorted(failure_mode_counts.items(), key=lambda item: (-item[1], item[0]))),
        "robustness_counts": robustness_counts,
        "rubric_averages": rubric_averages,
    }
    baseline_payload: dict[str, Any] | None = None
    gates_payload: dict[str, Any] | None = None
    if baseline_path and baseline_path.exists():
        loaded = _load_json(baseline_path)
        if loaded:
            baseline_payload = loaded
            payload["baseline_path"] = str(baseline_path)
    if gates_path and gates_path.exists():
        loaded = _load_json(gates_path)
        if loaded:
            gates_payload = loaded
            payload["gates_path"] = str(gates_path)
    if baseline_payload is None and gates_payload and isinstance(gates_payload.get("baseline_report"), str):
        candidate = Path(str(gates_payload.get("baseline_report", "")).strip())
        if candidate.exists():
            loaded = _load_json(candidate)
            if loaded:
                baseline_payload = loaded
                payload["baseline_path"] = str(candidate)
    payload["gate_evaluation"] = _evaluate_backlog_gates(
        current=payload,
        baseline=baseline_payload,
        gates=gates_payload,
    )
    output_path = report_out or _infer_metrics_report_path(ocr_out_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    payload["report_path"] = str(output_path)
    return payload


async def run_gemini_table_validation(
    *,
    ocr_out_dir: Path,
    config: GeminiValidationConfig,
    client: Any | None = None,
    only_problem_docs: bool = True,
    max_docs: int = 0,
    max_tables_per_doc: int = 0,
) -> dict[str, Any]:
    docs = _discover_ocr_doc_dirs(ocr_out_dir)
    summary: dict[str, Any] = {
        "ocr_out_dir": str(ocr_out_dir),
        "model": config.model,
        "docs_discovered": len(docs),
        "docs_considered": 0,
        "docs_reviewed": 0,
        "docs_skipped_ok": 0,
        "docs_skipped_missing_manifest": 0,
        "docs_skipped_missing_source_pdf": 0,
        "docs_skipped_no_tables": 0,
        "tables_reviewed": 0,
        "table_present_yes": 0,
        "table_present_no": 0,
        "table_present_uncertain": 0,
        "recommended_accept": 0,
        "recommended_review": 0,
        "recommended_reject": 0,
        "needs_followup_count": 0,
        "tables_with_llm_instructions": 0,
        "llm_instruction_count": 0,
        "failure_mode_counts": {},
        "robustness_counts": {
            "robust": 0,
            "mostly_robust": 0,
            "fragile": 0,
            "failed": 0,
        },
        "rubric_averages": {
            "formatting_fidelity": 0.0,
            "structural_fidelity": 0.0,
            "data_completeness": 0.0,
            "unit_symbol_fidelity": 0.0,
            "context_resolution": 0.0,
        },
        "api_error_count": 0,
    }
    if not docs:
        return summary

    rubric_sums = {
        "formatting_fidelity": 0.0,
        "structural_fidelity": 0.0,
        "data_completeness": 0.0,
        "unit_symbol_fidelity": 0.0,
        "context_resolution": 0.0,
    }

    owns_client = client is None
    review_client = client or genai.Client(api_key=config.api_key)
    docs_emitted = 0
    try:
        for doc_dir in docs:
            if only_problem_docs and not _is_problem_doc(doc_dir):
                summary["docs_skipped_ok"] += 1
                continue
            if max_docs > 0 and docs_emitted >= max_docs:
                break

            summary["docs_considered"] += 1
            manifest_path = doc_dir / "metadata" / "manifest.json"
            manifest = _load_json(manifest_path)
            if not manifest:
                summary["docs_skipped_missing_manifest"] += 1
                continue
            source_pdf = _resolve_source_pdf(manifest, doc_dir)
            if source_pdf is None:
                summary["docs_skipped_missing_source_pdf"] += 1
                continue

            tables_path = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
            tables = _load_jsonl(tables_path)
            if not tables:
                summary["docs_skipped_no_tables"] += 1
                continue
            if max_tables_per_doc > 0:
                tables = tables[:max_tables_per_doc]

            doc_rows: list[dict[str, Any]] = []
            page_cache: dict[int, bytes] = {}
            fragment_index = _resolve_fragment_index(doc_dir)
            with fitz.open(source_pdf) as pdf:
                for table in tables:
                    page = int(table.get("page", 0) or 0)
                    if page < 1 or page > int(pdf.page_count):
                        continue
                    validation_pages = _resolve_validation_pages(table, doc_dir)
                    pages: list[int] = []
                    for p in validation_pages or [page]:
                        if 1 <= int(p) <= int(pdf.page_count):
                            pages.append(int(p))
                    if not pages:
                        continue
                    image_bytes_list: list[bytes] = []
                    for p in pages:
                        if p not in page_cache:
                            page_cache[p] = _render_page_png(pdf, p, int(config.render_dpi))
                        image_bytes_list.append(page_cache[p])
                    context = _resolve_table_validation_context(
                        doc_dir=doc_dir,
                        table_row=table,
                        fragment_index=fragment_index,
                    )
                    crop_paths = _resolve_table_crop_paths(
                        doc_dir=doc_dir,
                        validation_pages=pages,
                        context=context,
                    )
                    crop_image_bytes_list: list[bytes] = []
                    for crop_path in crop_paths:
                        try:
                            crop_image_bytes_list.append(crop_path.read_bytes())
                        except Exception:
                            continue
                    table_for_prompt = dict(table)
                    table_for_prompt["validation_pages"] = pages
                    table_for_prompt["table_detail"] = context.get("table_detail", {})
                    table_for_prompt["extracted_csv"] = context.get("extracted_csv", "")
                    table_for_prompt["marker_fragments"] = context.get("marker_fragments", [])
                    table_for_prompt["ocr_html"] = context.get("ocr_html", "")
                    try:
                        review = await _review_table_with_gemini(
                            client=review_client,
                            config=config,
                            page_image_bytes_list=image_bytes_list,
                            crop_image_bytes_list=crop_image_bytes_list,
                            page_numbers=pages,
                            table_row=table_for_prompt,
                        )
                    except Exception as exc:  # noqa: BLE001
                        summary["api_error_count"] = int(summary.get("api_error_count", 0) or 0) + 1
                        review = _normalize_review(
                            {
                                "recommended_action": "review",
                                "needs_followup": True,
                                "issues": [f"gemini_api_error:{type(exc).__name__}"],
                                "failure_modes": ["other"],
                                "followup_recommendations": [
                                    "Retry Gemini validation for this table because the API call failed."
                                ],
                            },
                            raw_response="",
                        )
                        review["error"] = str(exc)
                    doc_rows.append(
                        {
                            "doc_dir": str(doc_dir),
                            "table_id": str(table.get("table_id", "")),
                            "page": page,
                            "validation_pages": pages,
                            "table_crop_count": len(crop_image_bytes_list),
                            "caption": str(table.get("caption", "")),
                            "headers": table.get("headers", []),
                            "rows": table.get("rows", []),
                            "model_review": review,
                        }
                    )
                    summary["tables_reviewed"] += 1
                    presence = str(review.get("table_present_on_page", "uncertain"))
                    if presence == "yes":
                        summary["table_present_yes"] += 1
                    elif presence == "no":
                        summary["table_present_no"] += 1
                    else:
                        summary["table_present_uncertain"] += 1
                    action = str(review.get("recommended_action", "review"))
                    if action == "accept":
                        summary["recommended_accept"] += 1
                    elif action == "reject":
                        summary["recommended_reject"] += 1
                    else:
                        summary["recommended_review"] += 1
                    if bool(review.get("needs_followup", False)):
                        summary["needs_followup_count"] += 1
                    llm_instructions = review.get("llm_extraction_instructions", [])
                    if isinstance(llm_instructions, list) and llm_instructions:
                        summary["tables_with_llm_instructions"] += 1
                        summary["llm_instruction_count"] += len(llm_instructions)
                    for mode in review.get("failure_modes", []):
                        key = str(mode).strip()
                        if not key:
                            continue
                        summary["failure_mode_counts"][key] = int(summary["failure_mode_counts"].get(key, 0)) + 1
                    rubric = review.get("rubric", {})
                    if isinstance(rubric, dict):
                        rob = str(rubric.get("overall_robustness", "")).strip()
                        if rob in summary["robustness_counts"]:
                            summary["robustness_counts"][rob] += 1
                        for rk in rubric_sums:
                            rubric_sums[rk] += _clamp01(rubric.get(rk, 0.0), default=0.0)

            if not doc_rows:
                continue
            docs_emitted += 1
            summary["docs_reviewed"] += 1
            validation_root = doc_dir / "metadata" / "assets" / "structured" / "validation"
            validation_root.mkdir(parents=True, exist_ok=True)
            report_path = validation_root / "gemini_table_review.jsonl"
            report_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in doc_rows) + "\n")
            doc_summary = {
                "doc_dir": str(doc_dir),
                "table_count": len(doc_rows),
                "table_present_yes": sum(1 for row in doc_rows if row["model_review"]["table_present_on_page"] == "yes"),
                "table_present_no": sum(1 for row in doc_rows if row["model_review"]["table_present_on_page"] == "no"),
                "table_present_uncertain": sum(
                    1 for row in doc_rows if row["model_review"]["table_present_on_page"] == "uncertain"
                ),
                "recommended_accept": sum(1 for row in doc_rows if row["model_review"]["recommended_action"] == "accept"),
                "recommended_review": sum(1 for row in doc_rows if row["model_review"]["recommended_action"] == "review"),
                "recommended_reject": sum(1 for row in doc_rows if row["model_review"]["recommended_action"] == "reject"),
                "needs_followup_count": sum(1 for row in doc_rows if bool(row["model_review"].get("needs_followup", False))),
                "tables_with_llm_instructions": sum(
                    1 for row in doc_rows if row["model_review"].get("llm_extraction_instructions", [])
                ),
                "llm_instruction_count": sum(
                    len(row["model_review"].get("llm_extraction_instructions", []))
                    for row in doc_rows
                    if isinstance(row["model_review"].get("llm_extraction_instructions", []), list)
                ),
                "failure_mode_counts": {},
                "robustness_counts": {
                    "robust": 0,
                    "mostly_robust": 0,
                    "fragile": 0,
                    "failed": 0,
                },
                "report_path": str(report_path),
            }
            for row in doc_rows:
                review = row.get("model_review", {})
                for mode in review.get("failure_modes", []):
                    key = str(mode).strip()
                    if not key:
                        continue
                    doc_summary["failure_mode_counts"][key] = int(doc_summary["failure_mode_counts"].get(key, 0)) + 1
                rubric = review.get("rubric", {})
                if isinstance(rubric, dict):
                    rob = str(rubric.get("overall_robustness", "")).strip()
                    if rob in doc_summary["robustness_counts"]:
                        doc_summary["robustness_counts"][rob] += 1
            (validation_root / "gemini_table_review_summary.json").write_text(
                json.dumps(doc_summary, indent=2, ensure_ascii=True)
            )
    finally:
        if owns_client and hasattr(review_client, "aio") and hasattr(review_client.aio, "aclose"):
            await review_client.aio.aclose()
        if owns_client and hasattr(review_client, "close"):
            review_client.close()

    reviewed = int(summary.get("tables_reviewed", 0) or 0)
    if reviewed > 0:
        for rk, value in rubric_sums.items():
            summary["rubric_averages"][rk] = round(float(value) / float(reviewed), 3)

    return summary
