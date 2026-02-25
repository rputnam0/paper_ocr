from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

GROUP_PAGE_TABLE_RE = re.compile(r"page_(\d+)_table_(\d+)$")


@dataclass
class TableStructureSummary:
    enabled: bool
    model: str
    command_configured: bool
    expected: int = 0
    generated: int = 0
    skipped_existing: int = 0
    missing_crop: int = 0
    failed: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    crops_root: str = ""
    report_path: str = ""

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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


def _bbox_from_polygons(polygons: list[list[list[float]]]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for poly in polygons:
        for point in poly:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except Exception:
                continue
    if not xs or not ys:
        return []
    return [min(xs), min(ys), max(xs), max(ys)]


def ensure_full_table_crops_for_doc(doc_dir: Path) -> dict[str, Any]:
    marker_tables_path = doc_dir / "metadata" / "assets" / "structured" / "marker" / "tables_raw.jsonl"
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    crops_root = qa_root / "bbox_table_crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(doc_dir / "metadata" / "manifest.json")
    source_path_raw = str(manifest.get("source_path", "")).strip()
    if not source_path_raw:
        return {
            "expected": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
            "status": "missing_source_path",
            "crops_root": _portable_path(crops_root, doc_dir),
        }
    source_path = Path(source_path_raw)
    if not source_path.exists():
        return {
            "expected": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
            "status": f"missing_source_pdf:{source_path}",
            "crops_root": _portable_path(crops_root, doc_dir),
        }
    table_rows = _load_jsonl(marker_tables_path)
    if not table_rows:
        return {
            "expected": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
            "status": "missing_tables_raw",
            "crops_root": _portable_path(crops_root, doc_dir),
        }

    expected = 0
    generated = 0
    skipped_existing = 0
    failed = 0
    by_page_counter: dict[int, int] = {}

    with fitz.open(source_path) as pdf:
        for row in table_rows:
            page = int(row.get("page", 0) or 0)
            if page < 1 or page > int(pdf.page_count):
                continue
            polygons = row.get("polygons")
            bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else []
            if (not isinstance(bbox, list) or len(bbox) < 4) and isinstance(polygons, list):
                bbox = _bbox_from_polygons(polygons)
            if not isinstance(bbox, list) or len(bbox) < 4:
                failed += 1
                continue
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if x1 <= x0 or y1 <= y0:
                failed += 1
                continue

            group = str(row.get("table_group_id", "")).strip()
            m = GROUP_PAGE_TABLE_RE.search(group)
            if m:
                ordinal = int(m.group(2))
            else:
                by_page_counter[page] = by_page_counter.get(page, 0) + 1
                ordinal = by_page_counter[page]

            expected += 1
            out_path = crops_root / f"table_{ordinal:02d}_page_{page:04d}.png"
            if out_path.exists():
                skipped_existing += 1
                continue

            try:
                page_obj = pdf.load_page(page - 1)
                clip = fitz.Rect(x0, y0, x1, y1)
                pix = page_obj.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=clip, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(out_path, format="PNG", optimize=True)
                generated += 1
            except Exception:
                failed += 1

    status = "ok" if failed == 0 else "partial"
    return {
        "expected": expected,
        "generated": generated,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "status": status,
        "crops_root": _portable_path(crops_root, doc_dir),
    }


def _table_group_to_crop_path(crops_root: Path, table_group_id: str) -> Path | None:
    m = GROUP_PAGE_TABLE_RE.search(str(table_group_id or "").strip())
    if not m:
        return None
    page = int(m.group(1))
    ordinal = int(m.group(2))
    out = crops_root / f"table_{ordinal:02d}_page_{page:04d}.png"
    return out if out.exists() else None


def _resolve_crop_for_table(doc_dir: Path, table_payload: dict[str, Any], fragment_index: dict[str, dict[str, Any]]) -> Path | None:
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    crops_root = qa_root / "bbox_table_crops"
    if not crops_root.exists():
        return None

    fragment_ids = table_payload.get("fragment_ids", [])
    if isinstance(fragment_ids, list):
        for item in fragment_ids:
            fid = str(item).strip()
            if not fid:
                continue
            frag = fragment_index.get(fid, {})
            crop_path = _table_group_to_crop_path(crops_root, str(frag.get("table_group_id", "")))
            if crop_path is not None:
                return crop_path

    row_lineage = table_payload.get("row_lineage", [])
    if isinstance(row_lineage, list):
        for item in row_lineage:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("fragment_id", "")).strip()
            if not fid:
                continue
            frag = fragment_index.get(fid, {})
            crop_path = _table_group_to_crop_path(crops_root, str(frag.get("table_group_id", "")))
            if crop_path is not None:
                return crop_path

    pages = table_payload.get("pages", [])
    if isinstance(pages, list):
        for item in pages:
            try:
                page = int(item)
            except Exception:
                continue
            candidates = sorted(crops_root.glob(f"table_*_page_{page:04d}.png"))
            if candidates:
                return candidates[0]
    return None


def _normalize_structure_payload(payload: dict[str, Any], *, model: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    def _as_int(*keys: str) -> int:
        for key in keys:
            raw = payload.get(key)
            try:
                value = int(raw)
            except Exception:
                continue
            if value > 0:
                return value
        return 0

    rows = _as_int("rows", "row_count", "num_rows", "n_rows", "total_rows")
    cols = _as_int("cols", "col_count", "num_cols", "n_cols", "total_cols")
    header_rows = _as_int("header_rows", "header_row_count")

    cells_raw = payload.get("cells", [])
    cells: list[dict[str, int]] = []
    if isinstance(cells_raw, list):
        for item in cells_raw:
            if not isinstance(item, dict):
                continue
            try:
                r0 = int(item.get("row_start", item.get("row0", item.get("row", 0))))
                r1 = int(item.get("row_end", item.get("row1", r0)))
                c0 = int(item.get("col_start", item.get("col0", item.get("col", 0))))
                c1 = int(item.get("col_end", item.get("col1", c0)))
            except Exception:
                continue
            if min(r0, r1, c0, c1) < 0:
                continue
            cells.append(
                {
                    "row_start": min(r0, r1),
                    "row_end": max(r0, r1),
                    "col_start": min(c0, c1),
                    "col_end": max(c0, c1),
                }
            )

    if rows <= 0 and cells:
        rows = max(cell["row_end"] for cell in cells) + 1
    if cols <= 0 and cells:
        cols = max(cell["col_end"] for cell in cells) + 1
    if rows <= 0 or cols <= 0:
        return {}

    html = str(payload.get("html_table", payload.get("html", "")) or "").strip()
    return {
        "model": model,
        "rows": rows,
        "cols": cols,
        "header_rows": header_rows,
        "cells": cells,
        "html_table": html,
    }


def _run_structure_command(*, command: str, image_path: Path, output_path: Path, timeout: int, model: str) -> tuple[dict[str, Any], str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = str(command).format(image=str(image_path), output=str(output_path))
    try:
        result = subprocess.run(
            rendered,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)

    payload: dict[str, Any] = {}
    if output_path.exists():
        payload = _load_json(output_path)
    elif str(result.stdout or "").strip():
        try:
            parsed = json.loads(str(result.stdout).strip())
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

    normalized = _normalize_structure_payload(payload, model=model)
    if not normalized:
        stderr = str(result.stderr or "").strip()
        return {}, stderr or "invalid_or_empty_structure_payload"
    return normalized, ""


def build_table_structure_artifacts(
    *,
    doc_dir: Path,
    model: str,
    command: str,
    timeout: int = 120,
) -> dict[str, Any]:
    enabled = str(model).strip().lower() != "off"
    command_configured = bool(str(command or "").strip())
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    structure_root = qa_root / "table_structure"
    structure_root.mkdir(parents=True, exist_ok=True)
    report_path = structure_root / "manifest.jsonl"
    summary = TableStructureSummary(
        enabled=enabled and command_configured,
        model=str(model),
        command_configured=command_configured,
        report_path=_portable_path(report_path, doc_dir),
        crops_root=_portable_path(qa_root / "bbox_table_crops", doc_dir),
    )

    if not enabled:
        _write_jsonl(report_path, [])
        return asdict(summary)
    if not command_configured:
        summary.errors.append("table_structure_command_missing")
        _write_jsonl(report_path, [])
        return asdict(summary)

    crop_summary = ensure_full_table_crops_for_doc(doc_dir)
    summary.crops_root = str(crop_summary.get("crops_root", summary.crops_root))
    if str(crop_summary.get("status", "")).strip().lower() not in {"ok", "partial"}:
        summary.errors.append(str(crop_summary.get("status", "crop_generation_failed")))
        _write_jsonl(report_path, [])
        return asdict(summary)

    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    table_manifest = _load_jsonl(tables_dir / "manifest.jsonl")
    fragment_index: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(doc_dir / "metadata" / "assets" / "structured" / "extracted" / "table_fragments.jsonl"):
        fid = str(row.get("fragment_id", "")).strip()
        if fid:
            fragment_index[fid] = row

    manifest_rows: list[dict[str, Any]] = []
    for row in table_manifest:
        table_id = str(row.get("table_id", "")).strip()
        if not table_id:
            continue
        summary.expected += 1
        table_payload = _load_json(tables_dir / f"{table_id}.json")
        crop_path = _resolve_crop_for_table(doc_dir, table_payload, fragment_index)
        if crop_path is None:
            summary.missing_crop += 1
            manifest_rows.append(
                {
                    "table_id": table_id,
                    "status": "missing_crop",
                    "model": model,
                }
            )
            continue

        structure_path = structure_root / f"{table_id}.json"
        if structure_path.exists():
            summary.skipped_existing += 1
            manifest_rows.append(
                {
                    "table_id": table_id,
                    "status": "ok",
                    "model": model,
                    "crop_path": _portable_path(crop_path, doc_dir),
                    "structure_path": _portable_path(structure_path, doc_dir),
                }
            )
            continue

        normalized, error = _run_structure_command(
            command=command,
            image_path=crop_path,
            output_path=structure_path,
            timeout=int(timeout),
            model=str(model),
        )
        if not normalized:
            summary.failed += 1
            if error:
                summary.errors.append(f"{table_id}:{error}")
            manifest_rows.append(
                {
                    "table_id": table_id,
                    "status": "failed",
                    "error": error,
                    "model": model,
                    "crop_path": _portable_path(crop_path, doc_dir),
                }
            )
            continue

        payload = {
            "table_id": table_id,
            "crop_path": _portable_path(crop_path, doc_dir),
            "status": "ok",
            **normalized,
        }
        structure_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        summary.generated += 1
        manifest_rows.append(
            {
                "table_id": table_id,
                "status": "ok",
                "model": model,
                "crop_path": _portable_path(crop_path, doc_dir),
                "structure_path": _portable_path(structure_path, doc_dir),
            }
        )

    _write_jsonl(report_path, manifest_rows)
    return asdict(summary)
