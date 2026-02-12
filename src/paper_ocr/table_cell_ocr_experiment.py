from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import request

import fitz
from openai import AsyncOpenAI
from PIL import Image

from .bibliography import extract_json_object
from .client import call_olmocr, call_text_model
from .structured_extract import run_marker_doc

GROUP_PAGE_TABLE_RE = re.compile(r"page_(\d+)_table_(\d+)$")
DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_OCR_MODEL = "allenai/olmOCR-2-7B-1025"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


@dataclass
class ExperimentRow:
    table_id: str
    table_group_id: str
    page: int
    status: str
    rows: int = 0
    cols: int = 0
    cell_count: int = 0
    error: str = ""
    result_path: str = ""
    llm_reconciled: bool = False
    llm_applied_corrections: bool = False


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out


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


def _is_normalized_bbox(raw_bbox: list[float]) -> bool:
    if len(raw_bbox) < 4:
        return False
    try:
        coords = [float(x) for x in raw_bbox[:4]]
    except Exception:
        return False
    return min(coords) >= -0.05 and max(coords) <= 1.5


def _coerce_bbox_pixels(
    raw_bbox: list[Any],
    *,
    crop_width: int,
    crop_height: int,
    bbox_mode: str,
) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    except Exception:
        return None

    mode = str(bbox_mode or "auto").strip().lower()
    if mode not in {"auto", "normalized", "pixels"}:
        mode = "auto"
    normalized = _is_normalized_bbox([x0, y0, x1, y1]) if mode == "auto" else mode == "normalized"

    if normalized:
        x0 *= float(crop_width)
        x1 *= float(crop_width)
        y0 *= float(crop_height)
        y1 *= float(crop_height)

    left = int(max(0, min(crop_width, math.floor(min(x0, x1)))))
    top = int(max(0, min(crop_height, math.floor(min(y0, y1)))))
    right = int(max(0, min(crop_width, math.ceil(max(x0, x1)))))
    bottom = int(max(0, min(crop_height, math.ceil(max(y0, y1)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _find_cell_bbox(item: dict[str, Any]) -> list[Any] | None:
    for key in ("bbox", "cell_bbox", "box", "rect", "coords"):
        raw = item.get(key)
        if isinstance(raw, list) and len(raw) >= 4:
            return raw
    return None


def _row_col_bounds(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        r0 = int(item.get("row_start", item.get("row0", item.get("row", 0))))
        r1 = int(item.get("row_end", item.get("row1", r0)))
        c0 = int(item.get("col_start", item.get("col0", item.get("col", 0))))
        c1 = int(item.get("col_end", item.get("col1", c0)))
    except Exception:
        return None
    if min(r0, r1, c0, c1) < 0:
        return None
    return (min(r0, r1), max(r0, r1), min(c0, c1), max(c0, c1))


def _pick_positive_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = _as_int(payload.get(key), 0)
        if value > 0:
            return value
    return 0


def _row_col_boxes_from_detections(payload: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    detections = payload.get("detections", [])
    if not isinstance(detections, list):
        return [], []
    row_boxes: list[list[float]] = []
    col_boxes: list[list[float]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label", "")).strip().lower()
        box = det.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        try:
            coords = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        except Exception:
            continue
        if label == "table row":
            row_boxes.append(coords)
        elif label == "table column":
            col_boxes.append(coords)
    row_boxes.sort(key=lambda b: (b[1], b[0]))
    col_boxes.sort(key=lambda b: (b[0], b[1]))
    return row_boxes, col_boxes


def _derive_bbox_from_row_col(
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    row_boxes: list[list[float]],
    col_boxes: list[list[float]],
) -> list[float] | None:
    if row_start < 0 or col_start < 0:
        return None
    if row_end >= len(row_boxes) or col_end >= len(col_boxes):
        return None
    top = min(row_boxes[i][1] for i in range(row_start, row_end + 1))
    bottom = max(row_boxes[i][3] for i in range(row_start, row_end + 1))
    left = min(col_boxes[j][0] for j in range(col_start, col_end + 1))
    right = max(col_boxes[j][2] for j in range(col_start, col_end + 1))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _normalize_structure_payload_with_bboxes(
    payload: dict[str, Any],
    *,
    crop_width: int,
    crop_height: int,
    bbox_mode: str = "auto",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    rows = _pick_positive_int(payload, ("rows", "row_count", "num_rows", "n_rows", "total_rows"))
    cols = _pick_positive_int(payload, ("cols", "col_count", "num_cols", "n_cols", "total_cols"))
    header_rows = _pick_positive_int(payload, ("header_rows", "header_row_count"))
    row_boxes, col_boxes = _row_col_boxes_from_detections(payload)

    cells_raw = payload.get("cells", payload.get("cell_boxes", []))
    cells: list[dict[str, Any]] = []
    if isinstance(cells_raw, list):
        for item in cells_raw:
            if not isinstance(item, dict):
                continue
            bounds = _row_col_bounds(item)
            if bounds is None:
                continue
            raw_bbox = _find_cell_bbox(item)
            row_start, row_end, col_start, col_end = bounds
            if raw_bbox is None and row_boxes and col_boxes:
                raw_bbox = _derive_bbox_from_row_col(
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    row_boxes=row_boxes,
                    col_boxes=col_boxes,
                )
            if raw_bbox is None:
                continue
            bbox_px = _coerce_bbox_pixels(
                raw_bbox,
                crop_width=crop_width,
                crop_height=crop_height,
                bbox_mode=bbox_mode,
            )
            if bbox_px is None:
                continue
            cell_payload: dict[str, Any] = {
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "bbox": bbox_px,
            }
            confidence = item.get("confidence")
            try:
                if confidence is not None:
                    cell_payload["confidence"] = float(confidence)
            except Exception:
                pass
            cells.append(cell_payload)

    if rows <= 0 and cells:
        rows = max(int(cell["row_end"]) for cell in cells) + 1
    if cols <= 0 and cells:
        cols = max(int(cell["col_end"]) for cell in cells) + 1
    if rows <= 0 or cols <= 0 or not cells:
        return {}

    return {
        "rows": rows,
        "cols": cols,
        "header_rows": header_rows,
        "cells": cells,
    }


def _bbox_with_padding(bbox: list[int], *, width: int, height: int, padding_px: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding_px))
    left = max(0, x0 - pad)
    top = max(0, y0 - pad)
    right = min(width, x1 + pad)
    bottom = min(height, y1 + pad)
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return [left, top, right, bottom]


def _apply_cell_text_to_grid(rows: int, cols: int, cells: list[dict[str, Any]]) -> list[list[str]]:
    grid = [["" for _ in range(max(cols, 0))] for _ in range(max(rows, 0))]
    for cell in cells:
        try:
            row_start = max(0, int(cell.get("row_start", 0)))
            row_end = max(0, int(cell.get("row_end", row_start)))
            col_start = max(0, int(cell.get("col_start", 0)))
            col_end = max(0, int(cell.get("col_end", col_start)))
        except Exception:
            continue
        text = str(cell.get("text", "")).strip()
        for row_idx in range(row_start, min(row_end + 1, rows)):
            for col_idx in range(col_start, min(col_end + 1, cols)):
                current = grid[row_idx][col_idx]
                if not current or len(text) > len(current):
                    grid[row_idx][col_idx] = text
    return grid


def _collapse_header_rows(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    width = max((len(row) for row in header_rows), default=0)
    out: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in header_rows:
            value = str(row[col] if col < len(row) else "").strip()
            if value:
                parts.append(value)
        out.append(" / ".join(parts))
    return out


def _grid_to_markdown(grid: list[list[str]], *, header_rows: int) -> str:
    if not grid:
        return ""
    ncols = max(len(row) for row in grid)
    padded = [row + [""] * (ncols - len(row)) for row in grid]
    header_count = max(1, min(header_rows if header_rows > 0 else 1, len(padded)))
    headers = _collapse_header_rows(padded[:header_count])
    body = padded[header_count:] if len(padded) > header_count else []

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _grid_to_sections(grid: list[list[str]], *, header_rows: int, expected_rows: int) -> tuple[list[list[str]], list[list[str]]]:
    ncols = max((len(row) for row in grid), default=0)
    header_count = max(1, int(header_rows) if int(header_rows) > 0 else 1)
    padded = [[str(c) for c in row] + [""] * (ncols - len(row)) for row in grid]
    headers = padded[:header_count]
    rows = padded[header_count:]
    while len(headers) < header_count:
        headers.append([""] * ncols)
    if expected_rows > 0:
        rows = rows[:expected_rows]
        while len(rows) < expected_rows:
            rows.append([""] * ncols)
    return headers, rows


def _build_llm_reconciliation_prompt(
    *,
    table_id: str,
    structure: dict[str, Any],
    prefilled_grid: list[list[str]],
    cell_ocr_rows: list[dict[str, Any]],
    full_table_ocr: str,
) -> str:
    request = {
        "table_id": table_id,
        "table_structure": {
            "rows": int(structure.get("rows", 0) or 0),
            "cols": int(structure.get("cols", 0) or 0),
            "header_rows": int(structure.get("header_rows", 0) or 0),
            "cells": [
                {
                    "row_start": int(cell.get("row_start", 0)),
                    "row_end": int(cell.get("row_end", 0)),
                    "col_start": int(cell.get("col_start", 0)),
                    "col_end": int(cell.get("col_end", 0)),
                }
                for cell in structure.get("cells", [])
                if isinstance(cell, dict)
            ],
        },
        "prefilled_grid_from_cell_ocr": prefilled_grid,
        "cell_level_ocr": cell_ocr_rows,
        "full_table_ocr": str(full_table_ocr or "")[:32000],
    }
    return (
        "You reconcile table extraction outputs.\n"
        "Return JSON only.\n"
        "Use table_structure as the topology lock for row/column layout.\n"
        "Prefer cell-level OCR values, but correct cutoffs/splits using full_table_ocr evidence.\n"
        "Do not hallucinate values not present in either OCR source.\n"
        "Required keys:\n"
        "- corrected_header_rows_full: list[list[str]]\n"
        "- corrected_rows: list[list[str]]\n"
        "- applied_corrections: boolean\n"
        "- notes: string\n\n"
        f"Input:\n{json.dumps(request, ensure_ascii=True)}"
    )


def _normalize_llm_reconciliation_payload(
    *,
    raw_payload: dict[str, Any],
    expected_rows: int,
    expected_cols: int,
    expected_header_rows: int,
    fallback_grid: list[list[str]],
) -> dict[str, Any]:
    header_count = max(1, int(expected_header_rows) if int(expected_header_rows) > 0 else 1)
    cols = max(1, int(expected_cols))
    fb_headers, fb_rows = _grid_to_sections(
        fallback_grid,
        header_rows=header_count,
        expected_rows=max(0, int(expected_rows)),
    )
    fb_headers = [(row + [""] * (cols - len(row)))[:cols] for row in fb_headers]
    fb_rows = [(row + [""] * (cols - len(row)))[:cols] for row in fb_rows]

    headers_raw = raw_payload.get("corrected_header_rows_full")
    rows_raw = raw_payload.get("corrected_rows")
    valid_headers = isinstance(headers_raw, list) and all(isinstance(row, list) for row in headers_raw)
    valid_rows = isinstance(rows_raw, list) and all(isinstance(row, list) for row in rows_raw)
    if not valid_headers or not valid_rows:
        out_headers = fb_headers
        out_rows = fb_rows
        return {
            "valid": False,
            "reason": "invalid_schema",
            "header_rows_full": out_headers,
            "rows": out_rows,
            "final_grid": [*out_headers, *out_rows],
            "applied_corrections": False,
            "notes": "invalid_llm_output; fallback_to_prefilled",
        }

    headers: list[list[str]] = []
    for row in headers_raw:
        headers.append([str(cell).strip() for cell in row][:cols] + [""] * max(0, cols - len(row)))
    rows: list[list[str]] = []
    for row in rows_raw:
        rows.append([str(cell).strip() for cell in row][:cols] + [""] * max(0, cols - len(row)))

    headers = headers[:header_count]
    while len(headers) < header_count:
        headers.append([""] * cols)
    target_rows = max(0, int(expected_rows))
    if target_rows > 0:
        rows = rows[:target_rows]
        while len(rows) < target_rows:
            rows.append([""] * cols)

    notes = str(raw_payload.get("notes", "")).strip()
    applied_corrections = bool(raw_payload.get("applied_corrections", False))
    return {
        "valid": True,
        "reason": "",
        "header_rows_full": headers,
        "rows": rows,
        "final_grid": [*headers, *rows],
        "applied_corrections": applied_corrections,
        "notes": notes,
    }


def _build_llm_json_repair_prompt(raw_text: str) -> str:
    return (
        "Convert the following output into one strict JSON object and output JSON only.\n"
        "Schema:\n"
        "{\n"
        '  "corrected_header_rows_full": [["..."]],\n'
        '  "corrected_rows": [["..."]],\n'
        '  "applied_corrections": true,\n'
        '  "notes": "string"\n'
        "}\n"
        "Do not include markdown fences.\n\n"
        f"Source output:\n{raw_text[:24000]}"
    )


def _bbox_from_marker_row(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            pass
    polygons = row.get("polygons")
    if isinstance(polygons, list):
        derived = _bbox_from_polygons(polygons)
        if len(derived) == 4:
            return derived
    return []


def _structure_command_payload(command: str, image_path: Path, output_path: Path, timeout: int) -> tuple[dict[str, Any], str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = str(command).format(image=str(image_path), output=str(output_path))
    try:
        proc = subprocess.run(
            rendered,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout),
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)

    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text())
        except Exception as exc:  # noqa: BLE001
            return {}, f"invalid_structure_json:{exc}"
        return parsed if isinstance(parsed, dict) else {}, ""

    stdout = str(proc.stdout or "").strip()
    if not stdout:
        stderr = str(proc.stderr or "").strip()
        return {}, stderr or "structure_command_no_output"
    try:
        parsed = json.loads(stdout)
    except Exception as exc:  # noqa: BLE001
        return {}, f"invalid_structure_stdout:{exc}"
    return (parsed if isinstance(parsed, dict) else {}), ""


def _multipart_upload_request(
    *,
    url: str,
    file_field: str,
    file_path: Path,
    content_type: str,
    form_fields: dict[str, str],
) -> request.Request:
    boundary = f"----paper-ocr-cell-exp-{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in form_fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request(url=url, method="POST", data=bytes(body))
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return req


def _unwrap_structure_service_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("structure", "result", "data", "prediction", "output"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return inner
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return inner[0]
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return {}


def _parse_key_value_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        token = str(raw).strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        k = key.strip()
        if k:
            out[k] = value.strip()
    return out


def _structure_service_payload(
    *,
    service_url: str,
    image_path: Path,
    output_path: Path,
    timeout: int,
    file_field: str,
    form_fields: dict[str, str],
) -> tuple[dict[str, Any], str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = _multipart_upload_request(
        url=service_url,
        file_field=file_field,
        file_path=image_path,
        content_type="image/png",
        form_fields=form_fields,
    )
    try:
        with request.urlopen(req, timeout=int(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return {}, f"invalid_structure_service_json:{exc}"

    payload = _unwrap_structure_service_payload(parsed)
    if not payload:
        return {}, "empty_structure_service_payload"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return payload, ""


def _table_group_to_id_map(doc_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    tables_dir = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables"
    for row in _load_jsonl(tables_dir / "manifest.jsonl"):
        table_id = str(row.get("table_id", "")).strip()
        if not table_id:
            continue
        payload = _load_json(tables_dir / f"{table_id}.json")
        table_group_id = str(payload.get("table_group_id", "")).strip().removeprefix("group:")
        if table_group_id:
            out[table_group_id] = table_id
    return out


def _fallback_table_id(group_id: str, page: int, bbox: list[float], index: int) -> str:
    seed = f"{group_id}|{page}|{bbox}|{index}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


async def _ocr_single_cell(
    *,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    crop: Image.Image,
) -> tuple[str, str]:
    prompt = (
        "Read this single table cell exactly. "
        "Preserve symbols, punctuation, superscripts/subscripts, and units. "
        "Return plain text only. If empty, return an empty string."
    )
    buf = BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    image_bytes = buf.getvalue()
    async with sem:
        try:
            response = await call_olmocr(
                client=client,
                model=model,
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type="image/png",
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)
    return str(response.content or "").strip(), ""


async def _ocr_cells_for_table(
    *,
    table_image: Image.Image,
    structure_cells: list[dict[str, Any]],
    cells_dir: Path,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    workers: int,
    padding_px: int,
) -> list[dict[str, Any]]:
    cells_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, int(workers)))
    width, height = table_image.size

    async def _task(cell: dict[str, Any], idx: int) -> dict[str, Any]:
        bbox = list(cell.get("bbox", []))
        if len(bbox) < 4:
            return {**cell, "text": "", "error": "missing_bbox"}
        padded = _bbox_with_padding(
            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            width=width,
            height=height,
            padding_px=padding_px,
        )
        crop = table_image.crop((padded[0], padded[1], padded[2], padded[3]))
        crop_path = cells_dir / (
            f"cell_{idx:04d}_r{int(cell.get('row_start', 0)):03d}-{int(cell.get('row_end', 0)):03d}"
            f"_c{int(cell.get('col_start', 0)):03d}-{int(cell.get('col_end', 0)):03d}.png"
        )
        crop.save(crop_path, format="PNG", optimize=True)
        text, error = await _ocr_single_cell(
            sem=sem,
            client=client,
            model=model,
            max_tokens=max_tokens,
            crop=crop,
        )
        return {
            **cell,
            "bbox_padded": padded,
            "crop_path": str(crop_path),
            "text": text,
            "error": error,
        }

    tasks = [_task(cell, idx) for idx, cell in enumerate(structure_cells, start=1)]
    return await asyncio.gather(*tasks)


async def _ocr_full_table(
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    table_image: Image.Image,
) -> tuple[str, str]:
    prompt = (
        "Extract this full table as one HTML table. "
        "Preserve headers, merged semantics, symbols, units, and all cell values. "
        "Return only one <table>...</table> block."
    )
    buf = BytesIO()
    table_image.save(buf, format="PNG", optimize=True)
    image_bytes = buf.getvalue()
    try:
        response = await call_olmocr(
            client=client,
            model=model,
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type="image/png",
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)
    return str(response.content or "").strip(), ""


def _compact_cell_rows_for_prompt(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in cells:
        out.append(
            {
                "row_start": int(cell.get("row_start", 0)),
                "row_end": int(cell.get("row_end", 0)),
                "col_start": int(cell.get("col_start", 0)),
                "col_end": int(cell.get("col_end", 0)),
                "bbox": [int(x) for x in list(cell.get("bbox", []))[:4]] if isinstance(cell.get("bbox"), list) else [],
                "text": str(cell.get("text", "")),
                "error": str(cell.get("error", "")),
            }
        )
    return out


async def _run_llm_reconciliation(
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    table_id: str,
    structure: dict[str, Any],
    prefilled_grid: list[list[str]],
    ocr_cells: list[dict[str, Any]],
    full_table_ocr: str,
    expected_rows: int,
    expected_cols: int,
    expected_header_rows: int,
) -> dict[str, Any]:
    async def _call_json_mode(prompt_text: str, token_limit: int) -> tuple[str, str]:
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=token_limit,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt_text}],
            )
            message = response.choices[0].message
            content = str(message.content or "").strip()
            if content:
                return content, ""
            return "", "empty_json_mode_content"
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    prompt = _build_llm_reconciliation_prompt(
        table_id=table_id,
        structure=structure,
        prefilled_grid=prefilled_grid,
        cell_ocr_rows=_compact_cell_rows_for_prompt(ocr_cells),
        full_table_ocr=full_table_ocr,
    )
    raw_text = ""
    repair_text = ""
    raw_payload: dict[str, Any] = {}
    error = ""
    try:
        raw_text, error = await _call_json_mode(prompt, int(max_tokens))
        if not raw_text:
            response = await call_text_model(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            raw_text = str(response.content or response.reasoning_content or "")
        raw_payload = extract_json_object(raw_text)
        if not raw_payload:
            repair_prompt = _build_llm_json_repair_prompt(raw_text)
            repair_text, repair_error = await _call_json_mode(
                repair_prompt,
                max(600, int(max_tokens // 2)),
            )
            if not repair_text:
                repair_response = await call_text_model(
                    client=client,
                    model=model,
                    prompt=repair_prompt,
                    max_tokens=max(600, int(max_tokens // 2)),
                )
                repair_text = str(repair_response.content or repair_response.reasoning_content or "")
                if repair_error and not error:
                    error = repair_error
            raw_payload = extract_json_object(repair_text)
            if not raw_payload:
                error = "empty_or_unparseable_json"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    normalized = _normalize_llm_reconciliation_payload(
        raw_payload=raw_payload,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        expected_header_rows=expected_header_rows,
        fallback_grid=prefilled_grid,
    )
    if error and normalized.get("valid"):
        normalized["valid"] = False
        normalized["reason"] = f"llm_call_failed:{error}"
        normalized["applied_corrections"] = False
        normalized["notes"] = str(normalized.get("notes", "") or f"llm_call_failed:{error}")

    return {
        "model": model,
        "prompt": prompt,
        "raw_text": raw_text,
        "repair_text": repair_text,
        "raw_payload": raw_payload,
        "error": error,
        "normalized": normalized,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


async def run_table_cell_ocr_experiment(args: argparse.Namespace) -> int:
    doc_dir = Path(args.doc_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        doc_dir / "metadata" / "assets" / "structured" / "qa" / "cell_ocr_experiment"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_command = str(args.table_structure_command or "").strip()
    structure_url = str(args.table_structure_url or "").strip()
    if not structure_command and not structure_url:
        raise SystemExit("Provide either --table-structure-url or --table-structure-command.")

    manifest = _load_json(doc_dir / "metadata" / "manifest.json")
    source_path_raw = str(manifest.get("source_path", "")).strip()
    if not source_path_raw:
        raise SystemExit(f"missing source_path in {doc_dir / 'metadata' / 'manifest.json'}")
    source_pdf = Path(source_path_raw)
    if not source_pdf.exists():
        raise SystemExit(f"source PDF not found: {source_pdf}")

    marker_url = str(args.marker_url or "").strip()
    if marker_url:
        marker_doc = run_marker_doc(
            pdf_path=source_pdf,
            marker_command="marker_single",
            timeout=int(args.marker_timeout),
            assets_root=doc_dir / "metadata" / "assets",
            profile="full_json",
            marker_url=marker_url,
        )
        if not marker_doc.success:
            raise SystemExit(f"marker service localization failed: {marker_doc.error}")

    marker_tables = _load_jsonl(doc_dir / "metadata" / "assets" / "structured" / "marker" / "tables_raw.jsonl")
    if not marker_tables:
        raise SystemExit("no marker table rows found")

    table_group_to_id = _table_group_to_id_map(doc_dir)
    requested_table_ids = {str(x).strip() for x in (args.table_id or []) if str(x).strip()}
    limit = max(0, int(args.limit))

    api_key = str(os.getenv("DEEPINFRA_API_KEY", "")).strip()
    if not args.dry_run and not api_key:
        raise SystemExit("DEEPINFRA_API_KEY is required unless --dry-run is used.")

    client = AsyncOpenAI(api_key=api_key, base_url=str(args.base_url)) if not args.dry_run else None
    run_rows: list[ExperimentRow] = []

    try:
        with fitz.open(source_pdf) as pdf:
            processed = 0
            for idx, row in enumerate(marker_tables, start=1):
                page = _as_int(row.get("page"), 0)
                if page < 1 or page > int(pdf.page_count):
                    continue
                bbox = _bbox_from_marker_row(row)
                if len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox
                if x1 <= x0 or y1 <= y0:
                    continue

                group_id = str(row.get("table_group_id", "")).strip()
                table_id = table_group_to_id.get(group_id) or _fallback_table_id(group_id, page, bbox, idx)
                if requested_table_ids and table_id not in requested_table_ids:
                    continue
                if limit and processed >= limit:
                    break

                table_dir = out_dir / table_id
                result_path = table_dir / "result.json"
                if bool(args.skip_existing) and result_path.exists():
                    run_rows.append(
                        ExperimentRow(
                            table_id=table_id,
                            table_group_id=group_id,
                            page=page,
                            status="skipped_existing",
                            result_path=_portable_path(result_path, out_dir),
                        )
                    )
                    processed += 1
                    continue

                table_dir.mkdir(parents=True, exist_ok=True)
                scale = float(args.render_dpi) / 72.0
                page_obj = pdf.load_page(page - 1)
                pix = page_obj.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    clip=fitz.Rect(x0, y0, x1, y1),
                    alpha=False,
                )
                table_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                table_image_path = table_dir / "table_crop.png"
                table_image.save(table_image_path, format="PNG", optimize=True)

                structure_raw_path = table_dir / "structure_raw.json"
                if structure_url:
                    structure_payload, structure_error = _structure_service_payload(
                        service_url=structure_url,
                        image_path=table_image_path,
                        output_path=structure_raw_path,
                        timeout=int(args.table_structure_timeout),
                        file_field=str(args.table_structure_file_field),
                        form_fields=_parse_key_value_items(list(args.table_structure_form or [])),
                    )
                else:
                    structure_payload, structure_error = _structure_command_payload(
                        structure_command,
                        table_image_path,
                        structure_raw_path,
                        int(args.table_structure_timeout),
                    )
                if not structure_payload:
                    run_rows.append(
                        ExperimentRow(
                            table_id=table_id,
                            table_group_id=group_id,
                            page=page,
                            status="failed_structure",
                            error=structure_error or "empty_structure_payload",
                        )
                    )
                    processed += 1
                    continue

                normalized = _normalize_structure_payload_with_bboxes(
                    structure_payload,
                    crop_width=table_image.width,
                    crop_height=table_image.height,
                    bbox_mode=str(args.bbox_mode),
                )
                if not normalized:
                    run_rows.append(
                        ExperimentRow(
                            table_id=table_id,
                            table_group_id=group_id,
                            page=page,
                            status="failed_structure_normalization",
                            error="no_valid_cell_bboxes",
                        )
                    )
                    processed += 1
                    continue

                cells_dir = table_dir / "cells"
                if args.dry_run:
                    ocr_cells = [{**cell, "text": "", "error": ""} for cell in normalized["cells"]]
                else:
                    assert client is not None
                    ocr_cells = await _ocr_cells_for_table(
                        table_image=table_image,
                        structure_cells=normalized["cells"],
                        cells_dir=cells_dir,
                        client=client,
                        model=str(args.ocr_model),
                        max_tokens=int(args.max_tokens),
                        workers=int(args.workers),
                        padding_px=int(args.cell_padding_px),
                    )

                prefilled_grid = _apply_cell_text_to_grid(
                    rows=int(normalized["rows"]),
                    cols=int(normalized["cols"]),
                    cells=ocr_cells,
                )
                prefilled_markdown = _grid_to_markdown(
                    prefilled_grid,
                    header_rows=int(normalized.get("header_rows", 1) or 1),
                )
                (table_dir / "table.md").write_text(prefilled_markdown + ("\n" if prefilled_markdown else ""))

                full_table_ocr_text = ""
                full_table_ocr_error = ""
                if not args.dry_run and bool(args.full_table_ocr):
                    assert client is not None
                    full_table_ocr_text, full_table_ocr_error = await _ocr_full_table(
                        client=client,
                        model=str(args.full_table_ocr_model),
                        max_tokens=int(args.full_table_ocr_max_tokens),
                        table_image=table_image,
                    )
                if full_table_ocr_text or full_table_ocr_error:
                    (table_dir / "full_table_ocr.md").write_text(
                        (
                            f"<!-- error: {full_table_ocr_error} -->\n" if full_table_ocr_error else ""
                        )
                        + f"{full_table_ocr_text}\n"
                    )

                llm_result: dict[str, Any] = {}
                final_grid = prefilled_grid
                final_header_rows = int(normalized.get("header_rows", 1) or 1)
                llm_reconciled = False
                llm_applied_corrections = False
                if not args.dry_run and bool(args.llm_reconcile):
                    assert client is not None
                    llm_result = await _run_llm_reconciliation(
                        client=client,
                        model=str(args.llm_model),
                        max_tokens=int(args.llm_max_tokens),
                        table_id=table_id,
                        structure=normalized,
                        prefilled_grid=prefilled_grid,
                        ocr_cells=ocr_cells,
                        full_table_ocr=full_table_ocr_text,
                        expected_rows=max(
                            0,
                            int(normalized["rows"]) - int(normalized.get("header_rows", 1) or 1),
                        ),
                        expected_cols=int(normalized["cols"]),
                        expected_header_rows=int(normalized.get("header_rows", 1) or 1),
                    )
                    normalized_out = llm_result.get("normalized", {})
                    if isinstance(normalized_out, dict):
                        final_grid = [list(row) for row in normalized_out.get("final_grid", prefilled_grid)]
                        llm_reconciled = bool(normalized_out.get("valid", False))
                        llm_applied_corrections = bool(normalized_out.get("applied_corrections", False))
                        llm_header_rows = _as_int(normalized.get("header_rows"), 1)
                        final_header_rows = max(1, llm_header_rows if llm_header_rows > 0 else 1)
                    _write_json(table_dir / "llm_reconciliation.json", llm_result)

                final_markdown = _grid_to_markdown(final_grid, header_rows=final_header_rows)
                (table_dir / "table_llm.md").write_text(final_markdown + ("\n" if final_markdown else ""))

                payload = {
                    "table_id": table_id,
                    "table_group_id": group_id,
                    "page": page,
                    "table_bbox_pdf": [x0, y0, x1, y1],
                    "table_crop_path": _portable_path(table_image_path, out_dir),
                    "structure_raw_path": _portable_path(structure_raw_path, out_dir),
                    "structure_source": "service" if structure_url else "command",
                    "rows": int(normalized["rows"]),
                    "cols": int(normalized["cols"]),
                    "header_rows": int(normalized.get("header_rows", 0) or 0),
                    "cells": ocr_cells,
                    "prefilled_grid": prefilled_grid,
                    "final_grid": final_grid,
                    "full_table_ocr": {
                        "text": full_table_ocr_text,
                        "error": full_table_ocr_error,
                        "path": _portable_path(table_dir / "full_table_ocr.md", out_dir)
                        if (table_dir / "full_table_ocr.md").exists()
                        else "",
                    },
                    "llm_reconciliation": {
                        "enabled": bool(args.llm_reconcile),
                        "reconciled": llm_reconciled,
                        "applied_corrections": llm_applied_corrections,
                        "path": _portable_path(table_dir / "llm_reconciliation.json", out_dir)
                        if (table_dir / "llm_reconciliation.json").exists()
                        else "",
                    },
                    "markdown_path": _portable_path(table_dir / "table.md", out_dir),
                    "markdown_llm_path": _portable_path(table_dir / "table_llm.md", out_dir),
                    "dry_run": bool(args.dry_run),
                    "bbox_mode": str(args.bbox_mode),
                }
                _write_json(result_path, payload)
                run_rows.append(
                    ExperimentRow(
                        table_id=table_id,
                        table_group_id=group_id,
                        page=page,
                        status="ok",
                        rows=int(normalized["rows"]),
                        cols=int(normalized["cols"]),
                        cell_count=len(normalized["cells"]),
                        result_path=_portable_path(result_path, out_dir),
                        llm_reconciled=llm_reconciled,
                        llm_applied_corrections=llm_applied_corrections,
                    )
                )
                processed += 1
    finally:
        if client is not None:
            await client.close()

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row.__dict__, ensure_ascii=True) + "\n" for row in run_rows),
    )
    summary = {
        "doc_dir": str(doc_dir),
        "source_pdf": str(source_pdf),
        "out_dir": str(out_dir),
        "requested_table_ids": sorted(requested_table_ids),
        "total": len(run_rows),
        "ok": sum(1 for row in run_rows if row.status == "ok"),
        "failed": sum(1 for row in run_rows if row.status.startswith("failed")),
        "skipped_existing": sum(1 for row in run_rows if row.status == "skipped_existing"),
        "llm_reconciled": sum(1 for row in run_rows if row.llm_reconciled),
        "llm_applied_corrections": sum(1 for row in run_rows if row.llm_applied_corrections),
        "dry_run": bool(args.dry_run),
        "manifest_path": str(manifest_path),
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary["failed"] == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-ocr-cell-ocr-experiment",
        description=(
            "Experimental Marker->table-structure->per-cell OCR pipeline. "
            "Runs outside the main paper-ocr flow."
        ),
    )
    parser.add_argument("doc_dir", type=Path, help="Path to a single output doc dir (doc_<id>).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for experiment artifacts (default: doc metadata qa/cell_ocr_experiment).",
    )
    parser.add_argument(
        "--table-id",
        action="append",
        default=[],
        help="Optional table_id filter; can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of tables to process.")
    parser.add_argument(
        "--table-structure-command",
        type=str,
        default="",
        help="External structure command template using {image} and {output}.",
    )
    parser.add_argument(
        "--table-structure-url",
        type=str,
        default=os.getenv("PAPER_OCR_TABLE_STRUCTURE_URL", ""),
        help="Table-structure service URL (WSL forwarded endpoint).",
    )
    parser.add_argument(
        "--table-structure-file-field",
        type=str,
        default="file",
        help="Multipart file field name for table-structure service.",
    )
    parser.add_argument(
        "--table-structure-form",
        action="append",
        default=[],
        help="Extra form fields for table-structure service (KEY=VALUE). Can be repeated.",
    )
    parser.add_argument("--table-structure-timeout", type=int, default=180)
    parser.add_argument("--bbox-mode", choices=["auto", "normalized", "pixels"], default="auto")
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--cell-padding-px", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Skip OCR calls; validate structure+bbox only.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--full-table-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one full-table OCR pass per table crop for reconciliation evidence.",
    )
    parser.add_argument(
        "--full-table-ocr-model",
        type=str,
        default=os.getenv("PAPER_OCR_TABLE_FULL_OCR_MODEL", DEFAULT_OCR_MODEL),
    )
    parser.add_argument(
        "--full-table-ocr-max-tokens",
        type=int,
        default=int(os.getenv("PAPER_OCR_TABLE_FULL_OCR_MAX_TOKENS", "1600")),
    )
    parser.add_argument(
        "--llm-reconcile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run final LLM reconciliation using structure + cell OCR + full table OCR.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("PAPER_OCR_TABLE_LLM_MODEL", DEFAULT_LLM_MODEL),
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=int(os.getenv("PAPER_OCR_TABLE_LLM_MAX_TOKENS", "1800")),
    )
    parser.add_argument(
        "--marker-url",
        type=str,
        default=os.getenv("PAPER_OCR_MARKER_URL", ""),
        help="Marker service URL (WSL forwarded endpoint). When provided, refreshes tables_raw.jsonl via service.",
    )
    parser.add_argument("--marker-timeout", type=int, default=180)
    parser.add_argument("--base-url", type=str, default=os.getenv("PAPER_OCR_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--ocr-model", type=str, default=DEFAULT_OCR_MODEL)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_table_cell_ocr_experiment(args))


if __name__ == "__main__":
    raise SystemExit(main())
