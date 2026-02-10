from __future__ import annotations

import base64
from html import unescape
from html.parser import HTMLParser
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request

import fitz

from .inspect import TextHeuristics, decide_route, is_structured_page_candidate


@dataclass
class StructuredPageResult:
    success: bool
    markdown: str = ""
    raw_json: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass
class GrobidResult:
    success: bool
    tei_xml: str = ""
    bibliography_patch: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    figures_tables: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass
class MarkerDocResult:
    success: bool
    markdown: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    localization_page_status: dict[int, dict[str, Any]] = field(default_factory=dict)
    error: str = ""


def is_structured_candidate_doc(
    digital_structured: str,
    routes: list[str],
    heuristics_by_page: list[TextHeuristics],
) -> bool:
    """Return True when a document should attempt Marker structured extraction.

    The `routes` list is validated for length but route values are intentionally
    recomputed in auto mode so `--mode` OCR overrides do not disable structured
    extraction eligibility.
    """
    mode = (digital_structured or "").strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    if mode != "auto":
        return False
    if not routes or not heuristics_by_page or len(routes) != len(heuristics_by_page):
        return False

    # Structured routing should use auto-mode route interpretation regardless
    # of OCR route overrides (`--mode anchored|unanchored`).
    auto_routes = [decide_route(h, mode="auto") for h in heuristics_by_page]
    candidate_ratio = sum(1 for h in heuristics_by_page if is_structured_page_candidate(h)) / len(heuristics_by_page)
    if candidate_ratio < 0.6:
        return False

    # Fast accept when title/first page is text-rich.
    if auto_routes[0] == "anchored":
        return True

    # Allow image-heavy cover pages when the body remains clearly born-digital.
    if len(auto_routes) < 3:
        return False
    body_heuristics = heuristics_by_page[1:]
    body_candidate_ratio = sum(1 for h in body_heuristics if is_structured_page_candidate(h)) / len(body_heuristics)
    return body_candidate_ratio >= 0.7


def normalize_markdown_for_llm(markdown: str) -> str:
    text = (markdown or "").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^(#{1,6})(\S)", r"\1 \2", text)
    text = re.sub(r"(?m)^(\s*[-*+])\s{2,}", r"\1 ", text)
    text = re.sub(r"(?im)^(figure|table)\s+(\d+)\s*[:.\-]?\s*(.+)$", lambda m: f"{m.group(1).capitalize()} {m.group(2)}: {m.group(3).strip()}", text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = "| " + " | ".join(cells) + " |"
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def merge_structured_page_markdown(page_markdowns: list[str]) -> str:
    chunks: list[str] = []
    for i, md in enumerate(page_markdowns, start=1):
        chunks.append(f"# Page {i}\n\n{normalize_markdown_for_llm(md)}")
    return "\n".join(chunks).strip() + "\n"


def _single_page_pdf(src_pdf: Path, page_index: int, dst_pdf: Path) -> None:
    with fitz.open(src_pdf) as src:
        with fitz.open() as out:
            out.insert_pdf(src, from_page=page_index, to_page=page_index)
            out.save(dst_pdf)


def _marker_commands(marker_command: str, input_pdf: Path, output_dir: Path) -> list[list[str]]:
    base = shlex.split(marker_command or "marker_single")
    if not base:
        base = ["marker_single"]
    if "--disable_ocr" not in base and not any(arg.startswith("--disable_ocr=") for arg in base):
        base = [*base, "--disable_ocr"]
    return [
        [*base, str(input_pdf), "--output_dir", str(output_dir), "--output_format", "markdown"],
        [*base, str(input_pdf), "--output_dir", str(output_dir)],
        [*base, str(input_pdf), str(output_dir)],
    ]


def _multipart_upload_request(
    *,
    url: str,
    file_field: str,
    file_path: Path,
    content_type: str,
    form_fields: dict[str, str] | None = None,
) -> request.Request:
    boundary = f"----paper-ocr-{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in (form_fields or {}).items():
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


def _extract_marker_output(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("markdown", "text", "content"):
            candidate = output.get(key)
            if isinstance(candidate, str):
                return candidate
    for key in ("markdown", "text", "content"):
        candidate = payload.get(key)
        if isinstance(candidate, str):
            return candidate
    return ""


def _write_marker_service_artifacts(
    *,
    payload: dict[str, Any],
    markdown: str,
    assets_root: Path,
    page_index: int,
) -> dict[str, str]:
    marker_root = assets_root / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    page_tag = f"page_{page_index + 1:04d}"

    md_path = marker_root / f"{page_tag}.md"
    md_path.write_text(markdown)

    image_names: list[str] = []
    images = payload.get("images")
    assets_dir: Path | None = None
    if isinstance(images, dict) and images:
        assets_dir = marker_root / f"{page_tag}_assets" / "service"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for key, encoded in images.items():
            if not isinstance(key, str) or not isinstance(encoded, str):
                continue
            name = Path(key).name or "image.jpeg"
            if "." not in name:
                name = f"{name}.jpeg"
            dst = assets_dir / name
            try:
                dst.write_bytes(base64.b64decode(encoded))
                image_names.append(name)
            except Exception:
                continue

    metadata = payload.get("metadata")
    metadata_obj = metadata if isinstance(metadata, dict) else {}
    json_path = marker_root / f"{page_tag}.json"
    json_path.write_text(
        json.dumps(
            {
                "source": "marker_service",
                "success": bool(payload.get("success", True)),
                "format": str(payload.get("format", "markdown")),
                "metadata": metadata_obj,
                "image_names": image_names,
                "output_chars": len(markdown),
            },
            indent=2,
            ensure_ascii=True,
        )
    )

    artifacts: dict[str, str] = {
        "markdown": str(md_path),
        "json": str(json_path),
    }
    if assets_dir is not None and image_names:
        artifacts["assets_dir"] = str(assets_dir.parent)
    return artifacts


def _run_marker_page_via_service(
    *,
    marker_url: str,
    single_pdf: Path,
    timeout: int,
    assets_root: Path | None,
    page_index: int,
) -> StructuredPageResult:
    endpoint = marker_url.rstrip("/") + "/marker/upload"
    req = _multipart_upload_request(
        url=endpoint,
        file_field="file",
        file_path=single_pdf,
        content_type="application/pdf",
        form_fields={
            "output_format": "markdown",
            "force_ocr": "false",
            "paginate_output": "false",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        loaded = json.loads(raw)
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:  # noqa: BLE001
        return StructuredPageResult(success=False, error=str(exc))

    if not bool(payload.get("success", True)):
        error = str(payload.get("error", "")).strip() or "Marker service reported failure."
        return StructuredPageResult(success=False, error=error)

    markdown = _extract_marker_output(payload)
    if not markdown.strip():
        return StructuredPageResult(success=False, error="Marker service returned no markdown output.")

    metadata = payload.get("metadata")
    raw_json = metadata if isinstance(metadata, dict) else {}
    artifacts: dict[str, str] = {}
    if assets_root is not None:
        artifacts = _write_marker_service_artifacts(
            payload=payload,
            markdown=markdown,
            assets_root=assets_root,
            page_index=page_index,
        )
    return StructuredPageResult(
        success=True,
        markdown=markdown,
        raw_json=raw_json,
        artifacts=artifacts,
    )


def write_structured_artifacts(
    marker_output_dir: Path,
    assets_root: Path,
    page_index: int,
    asset_level: str = "standard",
) -> dict[str, str]:
    marker_root = assets_root / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    page_tag = f"page_{page_index + 1:04d}"

    md_candidates = sorted(marker_output_dir.rglob("*.md"))
    if not md_candidates:
        return {}
    src_md = md_candidates[0]
    dst_md = marker_root / f"{page_tag}.md"
    shutil.copy2(src_md, dst_md)

    artifacts: dict[str, str] = {"markdown": str(dst_md)}

    json_candidates = sorted(marker_output_dir.rglob("*.json"))
    if json_candidates:
        src_json = json_candidates[0]
        dst_json = marker_root / f"{page_tag}.json"
        shutil.copy2(src_json, dst_json)
        artifacts["json"] = str(dst_json)

    copy_assets = asset_level == "full"
    if not copy_assets:
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        copy_assets = any(p.suffix.lower() in image_exts for p in marker_output_dir.rglob("*") if p.is_file())

    if copy_assets:
        assets_dir = marker_root / f"{page_tag}_assets"
        for p in marker_output_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() == ".md":
                continue
            if p.suffix.lower() == ".json" and asset_level != "full":
                continue
            if asset_level != "full" and p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            rel = p.relative_to(marker_output_dir)
            dst = assets_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        if assets_dir.exists():
            artifacts["assets_dir"] = str(assets_dir)
    return artifacts


def run_marker_page(
    pdf_path: Path,
    page_index: int,
    marker_command: str,
    timeout: int,
    assets_root: Path | None = None,
    asset_level: str = "standard",
    marker_url: str = "",
) -> StructuredPageResult:
    env = os.environ.copy()
    env["OCR_ENGINE"] = "None"
    last_error = ""

    with tempfile.TemporaryDirectory(prefix="paper_ocr_marker_") as tmp:
        tmp_dir = Path(tmp)
        single_pdf = tmp_dir / f"page_{page_index + 1:04d}.pdf"
        _single_page_pdf(pdf_path, page_index, single_pdf)

        if str(marker_url).strip():
            return _run_marker_page_via_service(
                marker_url=marker_url,
                single_pdf=single_pdf,
                timeout=timeout,
                assets_root=assets_root,
                page_index=page_index,
            )

        for cmd in _marker_commands(marker_command, single_pdf, tmp_dir / "out"):
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue

            md_candidates = sorted((tmp_dir / "out").rglob("*.md"))
            if not md_candidates:
                last_error = "Marker returned no markdown output."
                continue
            md_path = md_candidates[0]
            markdown = md_path.read_text()

            raw_json: dict[str, Any] | None = None
            json_candidates = sorted((tmp_dir / "out").rglob("*.json"))
            if json_candidates:
                try:
                    loaded = json.loads(json_candidates[0].read_text())
                    raw_json = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    raw_json = {}

            artifacts: dict[str, str] = {}
            if assets_root is not None:
                artifacts = write_structured_artifacts(
                    marker_output_dir=tmp_dir / "out",
                    assets_root=assets_root,
                    page_index=page_index,
                    asset_level=asset_level,
                )
            return StructuredPageResult(
                success=True,
                markdown=markdown,
                raw_json=raw_json,
                artifacts=artifacts,
            )

    return StructuredPageResult(success=False, error=last_error or "Marker extraction failed.")


def _multipart_request(url: str, pdf_path: Path) -> request.Request:
    return _multipart_upload_request(
        url=url,
        file_field="input",
        file_path=pdf_path,
        content_type="application/pdf",
        form_fields={"teiCoordinates": "figure"},
    )


def build_render_contract(
    *,
    pdf_page_w_pt: float,
    pdf_page_h_pt: float,
    render_page_w_px: int,
    render_page_h_px: int,
    rotation_degrees: int,
) -> dict[str, Any]:
    w_pt = max(float(pdf_page_w_pt), 1.0)
    h_pt = max(float(pdf_page_h_pt), 1.0)
    w_px = max(int(render_page_w_px), 1)
    h_px = max(int(render_page_h_px), 1)
    px_per_pt_x = w_px / w_pt
    px_per_pt_y = h_px / h_pt
    # Affine 2x3: x_px = a*x_pt + c*y_pt + e, y_px = b*x_pt + d*y_pt + f
    # Default no-rotation mapping in top-left y-down render space.
    pdf_to_px = [px_per_pt_x, 0.0, 0.0, px_per_pt_y, 0.0, 0.0]
    px_to_pdf = [1.0 / px_per_pt_x, 0.0, 0.0, 1.0 / px_per_pt_y, 0.0, 0.0]
    return {
        "pdf_page_w_pt": w_pt,
        "pdf_page_h_pt": h_pt,
        "render_page_w_px": w_px,
        "render_page_h_px": h_px,
        "dpi": round(px_per_pt_x * 72, 4),
        "px_per_pt_x": px_per_pt_x,
        "px_per_pt_y": px_per_pt_y,
        "origin_convention": "top_left",
        "y_axis_direction": "down",
        "rotation_degrees": int(rotation_degrees) % 360,
        "pdf_to_px_transform": pdf_to_px,
        "px_to_pdf_transform": px_to_pdf,
    }


def _apply_affine(x: float, y: float, matrix: list[float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def grobid_coords_to_px(
    coords: list[dict[str, Any]],
    *,
    page_contracts: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in coords:
        page = int(item.get("page") or 0)
        contract = page_contracts.get(page)
        if contract is None:
            continue
        matrix = contract.get("pdf_to_px_transform")
        if not isinstance(matrix, list) or len(matrix) != 6:
            continue
        try:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            w = float(item.get("w", 0.0))
            h = float(item.get("h", 0.0))
        except Exception:
            continue
        x0, y0 = _apply_affine(x, y, matrix)
        x1, y1 = _apply_affine(x + w, y + h, matrix)
        out.append(
            {
                "page": page,
                "x0": min(x0, x1),
                "y0": min(y0, y1),
                "x1": max(x0, x1),
                "y1": max(y0, y1),
            }
        )
    return out


def _marker_doc_commands(
    marker_command: str,
    input_pdf: Path,
    output_dir: Path,
    profile: str,
) -> list[list[str]]:
    base = shlex.split(marker_command or "marker_single")
    if not base:
        base = ["marker_single"]
    if "--disable_ocr" not in base and not any(arg.startswith("--disable_ocr=") for arg in base):
        base = [*base, "--disable_ocr"]
    output_format = "json" if profile == "full_json" else "markdown"
    return [
        [*base, str(input_pdf), "--output_dir", str(output_dir), "--output_format", output_format],
        [*base, str(input_pdf), "--output_dir", str(output_dir)],
        [*base, str(input_pdf), str(output_dir)],
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


PAGE_ID_RE = re.compile(r"/page/(\d+)/")
TABLE_NUM_RE = re.compile(r"\b(?:table|tab\.?)\s*([0-9ivxlcdm]+)\b", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def _marker_page_from_id(block_id: str) -> int | None:
    match = PAGE_ID_RE.search(str(block_id or ""))
    if not match:
        return None
    return int(match.group(1)) + 1


def _marker_bbox(block: dict[str, Any]) -> list[float]:
    bbox = block.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            return []
    return []


def _marker_polygons(block: dict[str, Any]) -> list[list[list[float]]]:
    poly = block.get("polygon")
    if not isinstance(poly, list) or not poly:
        return []
    # Marker commonly emits a single polygon [[x,y], ...]
    if isinstance(poly[0], list) and poly[0] and isinstance(poly[0][0], (int, float)):
        out = []
        pts = []
        for p in poly:
            if not isinstance(p, list) or len(p) < 2:
                continue
            try:
                pts.append([float(p[0]), float(p[1])])
            except Exception:
                continue
        if pts:
            out.append(pts)
        return out
    # Already polygon list [[[x,y],...], ...]
    out: list[list[list[float]]] = []
    for one in poly:
        if not isinstance(one, list):
            continue
        pts = []
        for p in one:
            if not isinstance(p, list) or len(p) < 2:
                continue
            try:
                pts.append([float(p[0]), float(p[1])])
            except Exception:
                continue
        if pts:
            out.append(pts)
    return out


def _plain_text_from_html(html_text: str) -> str:
    txt = str(html_text or "")
    txt = txt.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    txt = TAG_RE.sub(" ", txt)
    txt = unescape(txt)
    return " ".join(txt.split())


def _extract_table_number(text: str) -> str:
    m = TABLE_NUM_RE.search(str(text or ""))
    if not m:
        return ""
    return m.group(1).strip().lower()


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_tr = False
        self.in_cell = False
        self.cell_tag = ""
        self.cell_buf: list[str] = []
        self.cell_colspan = 1
        self.cell_rowspan = 1
        self.current_col = 0
        self.row_cells: list[tuple[int, str, str]] = []
        self.pending_rowspans: dict[int, tuple[int, str, str]] = {}
        self.header_rows: list[list[str]] = []
        self.data_rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self.in_tr = True
            self.row_cells = []
            self.current_col = 0
            self._consume_pending_spans()
            return
        if t in {"th", "td"} and self.in_tr:
            self._consume_pending_spans()
            self.in_cell = True
            self.cell_tag = t
            self.cell_buf = []
            self.cell_colspan = max(1, self._int_attr(attrs, "colspan", 1))
            self.cell_rowspan = max(1, self._int_attr(attrs, "rowspan", 1))
            return
        if t == "br" and self.in_cell:
            self.cell_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"th", "td"} and self.in_cell:
            text = unescape("".join(self.cell_buf)).strip()
            for _ in range(self.cell_colspan):
                self.row_cells.append((self.current_col, self.cell_tag, text))
                if self.cell_rowspan > 1:
                    self.pending_rowspans[self.current_col] = (self.cell_rowspan - 1, self.cell_tag, text)
                self.current_col += 1
            self.in_cell = False
            self.cell_tag = ""
            self.cell_buf = []
            self.cell_colspan = 1
            self.cell_rowspan = 1
            return
        if t == "tr" and self.in_tr:
            self._consume_pending_spans()
            if self.row_cells:
                max_col = max(c for c, _, _ in self.row_cells) + 1
                vals = [""] * max_col
                tags: list[str] = []
                for col, cell_tag, text in self.row_cells:
                    vals[col] = text
                    tags.append(cell_tag)
                if tags and all(k == "th" for k in tags):
                    self.header_rows.append(vals)
                else:
                    self.data_rows.append(vals)
            self.in_tr = False
            self.row_cells = []
            self.current_col = 0

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_buf.append(data)

    def _int_attr(self, attrs: list[tuple[str, str | None]], key: str, default: int) -> int:
        for name, value in attrs:
            if str(name).lower() != key:
                continue
            try:
                return int(str(value or default).strip())
            except Exception:
                return default
        return default

    def _consume_pending_spans(self) -> None:
        while self.current_col in self.pending_rowspans:
            remaining, cell_tag, text = self.pending_rowspans[self.current_col]
            self.row_cells.append((self.current_col, cell_tag, text))
            if remaining <= 1:
                self.pending_rowspans.pop(self.current_col, None)
            else:
                self.pending_rowspans[self.current_col] = (remaining - 1, cell_tag, text)
            self.current_col += 1


def _parse_html_table_rows(table_html: str) -> tuple[list[list[str]], list[list[str]]]:
    parser = _HTMLTableParser()
    parser.feed(str(table_html or ""))
    parser.close()
    all_rows = parser.header_rows + parser.data_rows
    max_cols = max((len(r) for r in all_rows), default=0)
    if max_cols:
        parser.header_rows = [list(r) + [""] * (max_cols - len(r)) for r in parser.header_rows]
        parser.data_rows = [list(r) + [""] * (max_cols - len(r)) for r in parser.data_rows]
    return parser.header_rows, parser.data_rows


def _flatten_marker_blocks(root_obj: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack: list[Any] = [root_obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "block_type" in node:
                block_id = str(node.get("id", ""))
                key = f"{block_id}|{node.get('block_type', '')}|{node.get('bbox', '')}"
                if key not in seen:
                    seen.add(key)
                    blocks.append(node)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, (dict, list)):
                    stack.append(value)
    return blocks


def _marker_rows_for_localization(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        block_type = str(block.get("block_type", "")).strip()
        if not block_type:
            continue
        page = _marker_page_from_id(str(block.get("id", "")))
        row: dict[str, Any] = {
            "type": block_type.lower(),
            "page": page or 0,
        }
        bbox = _marker_bbox(block)
        if bbox:
            row["bbox"] = bbox
        polygons = _marker_polygons(block)
        if polygons:
            row["polygon"] = polygons[0]
        rows.append(row)
    return rows


def _choose_caption_block(table_block: dict[str, Any], captions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not captions:
        return None
    table_html = str(table_block.get("html", "") or "")
    table_num = _extract_table_number(_plain_text_from_html(table_html))
    table_bbox = _marker_bbox(table_block)
    table_y = table_bbox[1] if table_bbox else 0.0
    matched: list[dict[str, Any]] = []
    for cap in captions:
        cap_num = _extract_table_number(_plain_text_from_html(str(cap.get("html", "") or "")))
        if table_num and cap_num and cap_num == table_num:
            matched.append(cap)
    candidates = matched or captions

    def _score(cap: dict[str, Any]) -> tuple[float, float]:
        cap_bbox = _marker_bbox(cap)
        if not cap_bbox:
            return (999999.0, 999999.0)
        cap_y = cap_bbox[1]
        vertical = abs(table_y - cap_y)
        above_penalty = 0.0 if cap_y <= table_y else 1000.0
        return (vertical + above_penalty, vertical)

    return sorted(candidates, key=_score)[0]


def _extract_tables_raw_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    captions_by_page: dict[int, list[dict[str, Any]]] = {}
    tables: list[dict[str, Any]] = []
    for block in blocks:
        btype = str(block.get("block_type", "")).lower()
        page = _marker_page_from_id(str(block.get("id", "")))
        if page is None:
            continue
        if btype == "caption":
            captions_by_page.setdefault(page, []).append(block)
        elif btype == "table":
            tables.append(block)

    tables_raw: list[dict[str, Any]] = []
    by_page_counter: dict[int, int] = {}
    for table_block in sorted(tables, key=lambda b: (_marker_page_from_id(str(b.get("id", ""))) or 0, str(b.get("id", "")))):
        page = _marker_page_from_id(str(table_block.get("id", "")))
        if page is None:
            continue
        by_page_counter[page] = by_page_counter.get(page, 0) + 1
        idx = by_page_counter[page]
        table_html = str(table_block.get("html", "") or "")
        header_rows, data_rows = _parse_html_table_rows(table_html)
        caption_block = _choose_caption_block(table_block, captions_by_page.get(page, []))
        caption_text = _plain_text_from_html(str((caption_block or {}).get("html", "")))
        if not caption_text:
            caption_text = _plain_text_from_html(table_html[:220])
        polygons = _marker_polygons(table_block)
        table_block_id = str(table_block.get("id", "") or "")
        caption_block_id = str((caption_block or {}).get("id", "") or "")
        tables_raw.append(
            {
                "table_group_id": f"page_{page:04d}_table_{idx:02d}",
                "table_block_ids": [table_block_id] if table_block_id else [],
                "caption_block_id": caption_block_id,
                "note_block_ids": [],
                "page": page,
                "polygons": polygons,
                "header_rows": header_rows,
                "data_rows": data_rows,
                "caption_text": caption_text,
                "caption_confidence": 0.95 if caption_block_id else 0.6,
                "source_format": "html",
                "html_table": table_html,
            }
        )
    return tables_raw


def _has_geometry(block: dict[str, Any]) -> bool:
    poly = block.get("polygon")
    if isinstance(poly, list) and poly:
        return True
    bbox = block.get("bbox")
    return isinstance(bbox, list) and len(bbox) >= 4


def _collect_localization_status(block_rows: list[dict[str, Any]], page_count: int) -> dict[int, dict[str, Any]]:
    status: dict[int, dict[str, Any]] = {}
    for page in range(1, page_count + 1):
        status[page] = {
            "has_candidate": False,
            "has_geometry": False,
            "layout_fallback_required": False,
        }
    for row in block_rows:
        page = int(row.get("page") or 0)
        if page < 1 or page > page_count:
            continue
        typ = str(row.get("type") or row.get("block_type") or "").lower()
        if typ in {"table", "figure", "caption"}:
            status[page]["has_candidate"] = True
            if _has_geometry(row):
                status[page]["has_geometry"] = True
    for page in range(1, page_count + 1):
        if status[page]["has_candidate"] and not status[page]["has_geometry"]:
            status[page]["layout_fallback_required"] = True
    return status


def run_marker_doc(
    pdf_path: Path,
    marker_command: str,
    timeout: int,
    assets_root: Path,
    profile: str = "full_json",
    marker_url: str = "",
) -> MarkerDocResult:
    marker_root = assets_root / "structured" / "marker"
    marker_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OCR_ENGINE"] = "None"
    last_error = ""

    with fitz.open(pdf_path) as doc:
        page_count = int(doc.page_count)

    with tempfile.TemporaryDirectory(prefix="paper_ocr_marker_doc_") as tmp:
        tmp_dir = Path(tmp)
        out_dir = tmp_dir / "out"

        if str(marker_url).strip():
            endpoint = marker_url.rstrip("/") + "/marker/upload"
            req = _multipart_upload_request(
                url=endpoint,
                file_field="file",
                file_path=pdf_path,
                content_type="application/pdf",
                form_fields={
                    "output_format": "json" if profile == "full_json" else "markdown",
                    "force_ocr": "false",
                    "paginate_output": "true",
                },
            )
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    payload = {}
                md = _extract_marker_output(payload)
                if md:
                    (marker_root / "full_document.md").write_text(md)
                (marker_root / "raw_service_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True))
                block_rows: list[dict[str, Any]] = [b for b in (payload.get("chunks") or payload.get("blocks") or []) if isinstance(b, dict)]
                tree_obj: Any | None = None
                output_obj = payload.get("output")
                if isinstance(output_obj, dict):
                    tree_obj = output_obj
                elif isinstance(output_obj, str):
                    try:
                        loaded_output = json.loads(output_obj)
                        if isinstance(loaded_output, dict):
                            tree_obj = loaded_output
                    except Exception:
                        tree_obj = None
                flat_blocks = _flatten_marker_blocks(tree_obj) if tree_obj is not None else []
                if flat_blocks:
                    derived_rows = _marker_rows_for_localization(flat_blocks)
                    if not block_rows:
                        block_rows = derived_rows
                    if derived_rows:
                        (marker_root / "blocks.jsonl").write_text(
                            "\n".join(json.dumps(r, ensure_ascii=True) for r in derived_rows) + "\n"
                        )
                    tables_raw = _extract_tables_raw_from_blocks(flat_blocks)
                    if tables_raw:
                        (marker_root / "tables_raw.jsonl").write_text(
                            "\n".join(json.dumps(r, ensure_ascii=True) for r in tables_raw) + "\n"
                        )
                if block_rows:
                    (marker_root / "blocks.jsonl").write_text(
                        "\n".join(json.dumps(r, ensure_ascii=True) for r in block_rows) + "\n"
                    )
                status = _collect_localization_status(block_rows, page_count=page_count)
                artifacts = {
                    "full_document_markdown": str(marker_root / "full_document.md"),
                    "raw_json": str(marker_root / "raw_service_payload.json"),
                }
                if (marker_root / "blocks.jsonl").exists():
                    artifacts["blocks"] = str(marker_root / "blocks.jsonl")
                if (marker_root / "tables_raw.jsonl").exists():
                    artifacts["tables_raw"] = str(marker_root / "tables_raw.jsonl")
                return MarkerDocResult(success=True, markdown=md, artifacts=artifacts, localization_page_status=status)
            except Exception as exc:  # noqa: BLE001
                return MarkerDocResult(success=False, error=str(exc))

        for cmd in _marker_doc_commands(marker_command, pdf_path, out_dir, profile):
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue

            md_candidates = sorted(out_dir.rglob("*.md"))
            markdown = md_candidates[0].read_text() if md_candidates else ""
            if markdown:
                shutil.copy2(md_candidates[0], marker_root / "full_document.md")

            json_candidates = sorted(out_dir.rglob("*.json"))
            raw_json_path = marker_root / "raw_doc.json"
            if json_candidates:
                shutil.copy2(json_candidates[0], raw_json_path)

            chunk_candidates = sorted(out_dir.rglob("*chunks*.jsonl"))
            block_rows: list[dict[str, Any]] = []
            if chunk_candidates:
                dst_chunks = marker_root / "chunks.jsonl"
                shutil.copy2(chunk_candidates[0], dst_chunks)
                block_rows = _read_jsonl(dst_chunks)
            if not block_rows and json_candidates:
                try:
                    loaded = json.loads(json_candidates[0].read_text())
                except Exception:
                    loaded = {}
                if isinstance(loaded, dict):
                    possible = loaded.get("chunks") or loaded.get("blocks") or loaded.get("items") or []
                    block_rows = [p for p in possible if isinstance(p, dict)]
                    if block_rows:
                        (marker_root / "blocks.jsonl").write_text(
                            "\n".join(json.dumps(r, ensure_ascii=True) for r in block_rows) + "\n"
                        )
                    flat_blocks = _flatten_marker_blocks(loaded)
                    if flat_blocks:
                        derived_rows = _marker_rows_for_localization(flat_blocks)
                        if not block_rows:
                            block_rows = derived_rows
                        if derived_rows and not (marker_root / "blocks.jsonl").exists():
                            (marker_root / "blocks.jsonl").write_text(
                                "\n".join(json.dumps(r, ensure_ascii=True) for r in derived_rows) + "\n"
                            )
                        tables_raw = _extract_tables_raw_from_blocks(flat_blocks)
                        if tables_raw:
                            (marker_root / "tables_raw.jsonl").write_text(
                                "\n".join(json.dumps(r, ensure_ascii=True) for r in tables_raw) + "\n"
                            )
            if not (marker_root / "blocks.jsonl").exists() and block_rows:
                (marker_root / "blocks.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=True) for r in block_rows) + "\n"
                )

            artifacts: dict[str, str] = {}
            if (marker_root / "full_document.md").exists():
                artifacts["full_document_markdown"] = str(marker_root / "full_document.md")
            if raw_json_path.exists():
                artifacts["raw_json"] = str(raw_json_path)
            if (marker_root / "chunks.jsonl").exists():
                artifacts["chunks"] = str(marker_root / "chunks.jsonl")
            if (marker_root / "blocks.jsonl").exists():
                artifacts["blocks"] = str(marker_root / "blocks.jsonl")
            if (marker_root / "tables_raw.jsonl").exists():
                artifacts["tables_raw"] = str(marker_root / "tables_raw.jsonl")

            status = _collect_localization_status(block_rows, page_count=page_count)
            return MarkerDocResult(
                success=True,
                markdown=markdown,
                artifacts=artifacts,
                localization_page_status=status,
            )

    return MarkerDocResult(success=False, error=last_error or "Marker document extraction failed.")


def _tei_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _coord_page(value: str) -> int | None:
    token = str(value).strip()
    if not token:
        return None
    try:
        return int(float(token))
    except Exception:
        return None


def _parse_coords_string(raw: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in str(raw or "").split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) < 5:
            continue
        page = _coord_page(parts[0])
        try:
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except Exception:
            continue
        out.append(
            {
                "raw": item,
                "page": page,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
    return out


def _parse_figures_tables(
    root: ET.Element,
    ns: dict[str, str],
    doc_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fig in root.findall(".//tei:figure", ns):
        fig_type = str(fig.attrib.get("type", "figure")).strip().lower()
        if fig_type != "table":
            fig_type = "figure"
        label = _tei_text(fig.find("tei:label", ns))
        head = _tei_text(fig.find("tei:head", ns))
        fig_desc = _tei_text(fig.find("tei:figDesc", ns))
        caption_parts = [p for p in (head, fig_desc) if p]
        caption_text = " ".join(caption_parts).strip()

        coord_strings: list[str] = []
        for node in fig.iter():
            raw = str(node.attrib.get("coords", "")).strip()
            if raw:
                coord_strings.append(raw)
        dedup_coords = list(dict.fromkeys(coord_strings))
        coords: list[dict[str, Any]] = []
        for coord_raw in dedup_coords:
            coords.extend(_parse_coords_string(coord_raw))

        page = next((int(c["page"]) for c in coords if c.get("page") is not None), None)
        if page is None:
            pb = fig.find(".//tei:pb", ns)
            if pb is not None:
                page = _coord_page(pb.attrib.get("n", ""))

        if not (label or caption_text or coords):
            continue
        records.append(
            {
                "doc_id": doc_id,
                "type": fig_type,
                "label": label,
                "caption_text": caption_text,
                "page": page,
                "coords": coords,
            }
        )
    return records


def _parse_grobid_tei(tei_xml: str, doc_id: str = "") -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei_xml)

    title = root.findtext(".//tei:titleStmt/tei:title", default="", namespaces=ns).strip()
    authors: list[str] = []
    for author in root.findall(".//tei:titleStmt/tei:author", ns):
        surname = author.findtext(".//tei:surname", default="", namespaces=ns).strip()
        forename_nodes = author.findall(".//tei:forename", ns)
        given_parts = [str(node.text or "").strip() for node in forename_nodes if str(node.text or "").strip()]
        given = " ".join(given_parts)
        if surname and given:
            authors.append(f"{surname}, {given}")
        elif surname:
            authors.append(surname)
        else:
            txt = _tei_text(author)
            if txt:
                authors.append(txt)

    year = ""
    date_el = root.find(".//tei:publicationStmt//tei:date", ns)
    if date_el is not None:
        when = (date_el.attrib.get("when", "") or "").strip()
        year_match = re.search(r"\b(19|20)\d{2}\b", when or _tei_text(date_el))
        if year_match:
            year = year_match.group(0)

    sections: list[dict[str, Any]] = []
    for div in root.findall(".//tei:text/tei:body//tei:div", ns):
        head = div.find("tei:head", ns)
        sec_title = _tei_text(head)
        if not sec_title:
            continue
        pages: list[int] = []
        for pb in div.findall(".//tei:pb", ns):
            n = str(pb.attrib.get("n", "")).strip()
            if n.isdigit():
                pages.append(int(n))
        if pages:
            start_page = min(pages)
            end_page = max(pages)
        else:
            start_page = 1
            end_page = 1
        sections.append(
            {
                "title": sec_title,
                "start_page": start_page,
                "end_page": end_page,
                "summary": "",
            }
        )

    patch = {
        "title": title,
        "authors": authors,
        "year": year,
    }
    figures_tables = _parse_figures_tables(root, ns, doc_id)
    return patch, sections, figures_tables


def run_grobid_doc(
    pdf_path: Path,
    grobid_url: str,
    timeout: int,
    tei_out_path: Path,
    doc_id: str = "",
) -> GrobidResult:
    if not str(grobid_url).strip():
        return GrobidResult(success=False, error="No GROBID URL configured.")
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"

    try:
        req = _multipart_request(endpoint, pdf_path)
        with request.urlopen(req, timeout=timeout) as resp:
            tei_xml = resp.read().decode("utf-8", errors="replace")
        tei_out_path.parent.mkdir(parents=True, exist_ok=True)
        tei_out_path.write_text(tei_xml)
        patch, sections, figures_tables = _parse_grobid_tei(tei_xml, doc_id=doc_id)
        figures_tables_path = tei_out_path.parent / "figures_tables.jsonl"
        if figures_tables:
            figures_tables_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in figures_tables) + "\n"
            )
        else:
            figures_tables_path.write_text("")
        return GrobidResult(
            success=True,
            tei_xml=tei_xml,
            bibliography_patch=patch,
            sections=sections,
            figures_tables=figures_tables,
            artifacts={
                "tei_xml": str(tei_out_path),
                "figures_tables": str(figures_tables_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return GrobidResult(success=False, error=str(exc))
