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

    return {
        "table_present_on_page": presence,
        "extraction_quality": quality,
        "false_positive_risk": risk,
        "recommended_action": action,
        "issues": issues,
        "raw_response": raw_response,
    }


def _short(value: Any, *, limit: int = 80) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_review_prompt(table_row: dict[str, Any]) -> str:
    raw_headers = table_row.get("headers", [])
    headers = [_short(x, limit=56) for x in raw_headers[:8]] if isinstance(raw_headers, list) else []
    raw_rows = table_row.get("rows", [])
    first_row: list[str] = []
    if isinstance(raw_rows, list) and raw_rows and isinstance(raw_rows[0], list):
        first_row = [_short(cell, limit=56) for cell in raw_rows[0][:8]]

    return (
        "Task: Validate one extracted table candidate against the page image.\n"
        "Return JSON only (no markdown) with keys exactly:\n"
        'table_present_on_page ("yes"|"no"|"uncertain"),\n'
        "extraction_quality (0..1),\n"
        "false_positive_risk (0..1),\n"
        'recommended_action ("accept"|"review"|"reject"),\n'
        "issues (string array).\n"
        "Scoring guidance:\n"
        "- Accept only when table structure/cells are mostly correct.\n"
        "- Review when table exists but extraction has notable issues.\n"
        "- Reject when table candidate is mostly wrong.\n"
        "Candidate summary:\n"
        f"table_id={_short(table_row.get('table_id', ''), limit=32)}\n"
        f"page={int(table_row.get('page', 0) or 0)}\n"
        f"caption={_short(table_row.get('caption', ''), limit=180)}\n"
        f"headers={' | '.join(headers)}\n"
        f"first_row={' | '.join(first_row)}"
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
    image_bytes: bytes,
    table_row: dict[str, Any],
) -> dict[str, Any]:
    model = _canonical_model_id(config.model)
    prompt = _build_review_prompt(table_row)

    # Test clients can provide a lightweight async method with the same semantic contract.
    if hasattr(client, "generate_table_review"):
        raw_response = await client.generate_table_review(
            model=model,
            prompt=prompt,
            image_bytes=image_bytes,
            generation_config=_generate_config(config),
        )
        parsed = _extract_json_object(str(raw_response))
        review = _normalize_review(parsed, raw_response=str(raw_response))
        review["model_used"] = model
        return review

    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
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
            with fitz.open(source_pdf) as pdf:
                for table in tables:
                    page = int(table.get("page", 0) or 0)
                    if page < 1 or page > int(pdf.page_count):
                        continue
                    if page not in page_cache:
                        page_cache[page] = _render_page_png(pdf, page, int(config.render_dpi))
                    review = await _review_table_with_gemini(
                        client=review_client,
                        config=config,
                        image_bytes=page_cache[page],
                        table_row=table,
                    )
                    doc_rows.append(
                        {
                            "doc_dir": str(doc_dir),
                            "table_id": str(table.get("table_id", "")),
                            "page": page,
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
