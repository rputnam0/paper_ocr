from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REQUIRED_FIELDS = {
    "record_id",
    "doc_id",
    "material_name_raw",
    "property_name_raw",
    "value_raw",
    "unit_raw",
    "source",
    "confidence",
    "extraction_stage",
    "created_at",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if rows:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")
    else:
        path.write_text("")


def _stable_id(*parts: str) -> str:
    payload = "::".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _extract_unit(header: str) -> str:
    text = str(header or "")
    match = re.search(r"\(([^)]+)\)", text)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _extract_numeric(value: str) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _table_records(doc_id: str, table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    created_at = _utc_now_iso()
    for table in table_rows:
        table_id = str(table.get("table_id", ""))
        page = int(table.get("page", 0) or 0)
        raw_headers = table.get("headers", [])
        headers = raw_headers if isinstance(raw_headers, list) else []
        rows = [list(r) for r in table.get("rows", []) if isinstance(r, list)]
        if len(headers) < 2:
            continue
        for row_idx, row in enumerate(rows):
            material = str(row[0]).strip() if len(row) > 0 else ""
            for col_idx in range(1, min(len(headers), len(row))):
                prop = str(headers[col_idx] if headers[col_idx] is not None else "").strip()
                if not prop:
                    continue
                value_raw = str(row[col_idx]).strip()
                unit = _extract_unit(prop)
                value_num = _extract_numeric(value_raw)
                confidence = 0.55
                if material:
                    confidence += 0.1
                if value_num is not None:
                    confidence += 0.2
                if unit:
                    confidence += 0.1
                confidence = max(0.0, min(confidence, 0.99))
                record = {
                    "record_id": _stable_id(doc_id, table_id, str(row_idx), str(col_idx), value_raw),
                    "doc_id": doc_id,
                    "material_name_raw": material,
                    "material_name_normalized": material.lower() if material else "",
                    "property_name_raw": prop,
                    "property_name_normalized": prop.lower(),
                    "value_raw": value_raw,
                    "value_numeric": value_num,
                    "unit_raw": unit,
                    "unit_normalized": unit.lower() if unit else "",
                    "conditions": {},
                    "method_context": "table",
                    "source": {
                        "kind": "table_cell",
                        "page": page,
                        "table_id": table_id,
                        "row_index": row_idx,
                        "col_index": col_idx,
                        "caption": str(table.get("caption", "")),
                    },
                    "confidence": round(float(confidence), 4),
                    "extraction_stage": "table_v1",
                    "created_at": created_at,
                }
                records.append(record)
    return records


def _figure_records(doc_id: str, figure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    created_at = _utc_now_iso()
    for idx, figure in enumerate(figure_rows):
        figure_id = str(figure.get("figure_id", "")) or f"figure_{idx+1}"
        alt_text = str(figure.get("alt_text", "")).strip()
        image_ref = str(figure.get("image_ref", "")).strip()
        page = int(figure.get("page", 0) or 0)
        payload = alt_text or image_ref
        record = {
            "record_id": _stable_id(doc_id, figure_id, payload),
            "doc_id": doc_id,
            "material_name_raw": "",
            "material_name_normalized": "",
            "property_name_raw": "figure_observation",
            "property_name_normalized": "figure_observation",
            "value_raw": payload,
            "value_numeric": None,
            "unit_raw": "",
            "unit_normalized": "",
            "conditions": {},
            "method_context": "figure",
            "source": {
                "kind": "figure",
                "page": page,
                "figure_id": figure_id,
                "image_ref": image_ref,
            },
            "confidence": 0.3,
            "extraction_stage": "figure_v1",
            "created_at": created_at,
        }
        records.append(record)
    return records


def validate_property_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in sorted(REQUIRED_FIELDS):
        if field not in record:
            errors.append(f"missing:{field}")
    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("invalid:source")
    confidence = record.get("confidence")
    try:
        c = float(confidence)
    except Exception:
        errors.append("invalid:confidence")
    else:
        if c < 0.0 or c > 1.0:
            errors.append("invalid:confidence_range")
    return errors


def export_facts_for_doc(doc_dir: Path) -> dict[str, Any]:
    metadata_dir = doc_dir / "metadata"
    manifest_path = metadata_dir / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text())
            manifest = payload if isinstance(payload, dict) else {}
        except Exception:
            manifest = {}
    else:
        manifest = {}
    doc_id = str(manifest.get("doc_id", "")).strip() or doc_dir.name.replace("doc_", "")

    tables_manifest = metadata_dir / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    figures_manifest = metadata_dir / "assets" / "structured" / "extracted" / "figures" / "manifest.jsonl"
    table_rows = _load_jsonl(tables_manifest)
    figure_rows = _load_jsonl(figures_manifest)

    records: list[dict[str, Any]] = []
    records.extend(_table_records(doc_id, table_rows))
    records.extend(_figure_records(doc_id, figure_rows))

    validation_errors: list[str] = []
    for rec in records:
        errs = validate_property_record(rec)
        if errs:
            validation_errors.append(f"{rec.get('record_id', '')}:{','.join(errs)}")

    facts_root = metadata_dir / "assets" / "structured" / "facts"
    records_path = facts_root / "property_records.jsonl"
    manifest_out = facts_root / "manifest.json"
    _write_jsonl(records_path, records)
    facts_manifest = {
        "doc_id": doc_id,
        "record_count": len(records),
        "errors": validation_errors,
        "records_path": "metadata/assets/structured/facts/property_records.jsonl",
        "schema_path": "docs/schemas/property_record.schema.json",
    }
    facts_root.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(facts_manifest, indent=2, ensure_ascii=True))

    manifest["facts_extraction"] = {
        "enabled": True,
        "record_count": len(records),
        "errors": validation_errors,
        "records_path": "metadata/assets/structured/facts/property_records.jsonl",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    return facts_manifest
