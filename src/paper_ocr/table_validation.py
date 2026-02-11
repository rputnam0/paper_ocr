from __future__ import annotations

import json
import re
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
                    "symbol_or_unit_loss",
                    "caption_mismatch",
                    "code_legend_unresolved",
                    "table_not_present",
                    "ocr_noise",
                    "other",
                ],
            },
        },
        "root_cause_hypothesis": {"type": "string"},
        "needs_followup": {"type": "boolean"},
        "followup_recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "table_present_on_page",
        "extraction_quality",
        "false_positive_risk",
        "recommended_action",
        "issues",
    ],
}


@dataclass
class GeminiValidationConfig:
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    max_output_tokens: int = 320
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
        "Judge whether final extraction preserves structure, headers, rows, symbols/units, and continuation semantics.\n"
        "If table uses encoded labels/codes not resolved in the extracted output, flag code_legend_unresolved.\n"
        "Return JSON only (no markdown) with keys exactly:\n"
        'table_present_on_page ("yes"|"no"|"uncertain"),\n'
        "extraction_quality (0..1),\n"
        "false_positive_risk (0..1),\n"
        'recommended_action ("accept"|"review"|"reject"),\n'
        "issues (string array),\n"
        "failure_modes (string array),\n"
        "root_cause_hypothesis (string),\n"
        "needs_followup (boolean),\n"
        "followup_recommendations (string array).\n"
        "Scoring guidance:\n"
        "- Accept only when table structure/cells are mostly correct.\n"
        "- Review when table exists but extraction has notable issues.\n"
        "- Reject when table candidate is mostly wrong.\n"
        "Failure mode taxonomy:\n"
        "- split_table_continuation, header_misalignment, row_shift_or_merge, column_shift_or_merge,\n"
        "  symbol_or_unit_loss, caption_mismatch, code_legend_unresolved, table_not_present, ocr_noise, other.\n"
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
    }
    if not docs:
        return summary

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
                    review = await _review_table_with_gemini(
                        client=review_client,
                        config=config,
                        page_image_bytes_list=image_bytes_list,
                        crop_image_bytes_list=crop_image_bytes_list,
                        page_numbers=pages,
                        table_row=table_for_prompt,
                    )
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
                "report_path": str(report_path),
            }
            (validation_root / "gemini_table_review_summary.json").write_text(
                json.dumps(doc_summary, indent=2, ensure_ascii=True)
            )
    finally:
        if owns_client and hasattr(review_client, "aio") and hasattr(review_client.aio, "aclose"):
            await review_client.aio.aclose()
        if owns_client and hasattr(review_client, "close"):
            review_client.close()

    return summary
