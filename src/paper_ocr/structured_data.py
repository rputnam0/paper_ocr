from __future__ import annotations

import csv
import hashlib
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FIGURE_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
OCR_TABLE_FILE_RE = re.compile(r"table_(\d+)_page_(\d+)\.md$", re.IGNORECASE)
SYMBOL_CHAR_RE = re.compile(r"[^\x00-\x7F]|[±≤≥≈×÷°µμδΔητσγβαΩω]")


@dataclass
class StructuredExportSummary:
    table_count: int = 0
    figure_count: int = 0
    deplot_count: int = 0
    unresolved_figure_count: int = 0
    errors: list[str] = field(default_factory=list)
    ocr_merge: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableFragment:
    fragment_id: str
    table_group_id: str
    explicit_group_id: bool
    table_block_ids: list[str]
    caption_block_id: str
    note_block_ids: list[str]
    page: int
    polygons: list[list[list[float]]]
    bbox: list[float]
    header_rows: list[list[str]]
    data_rows: list[list[str]]
    caption_text: str
    caption_confidence: float
    source_format: str
    quality_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAFlag:
    flag_id: str
    severity: str
    type: str
    page: int
    table_ref: str
    details: str


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    # Remove one leading/trailing table delimiter but keep empty edge cells.
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(line: str, expected_cols: int) -> bool:
    cells = _split_table_row(line)
    if len(cells) != expected_cols or expected_cols == 0:
        return False
    for cell in cells:
        if not re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")):
            return False
    return True


def _table_caption(lines: list[str], header_line_index: int) -> str:
    for idx in range(header_line_index - 1, -1, -1):
        candidate = lines[idx].strip()
        if not candidate:
            continue
        if candidate.lower().startswith("table "):
            return candidate
        return ""
    return ""


def extract_markdown_tables(markdown: str) -> list[dict[str, Any]]:
    lines = (markdown or "").splitlines()
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(lines) - 1:
        header = _split_table_row(lines[i])
        if not header:
            i += 1
            continue
        if not _is_table_separator(lines[i + 1], len(header)):
            i += 1
            continue

        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines):
            row = _split_table_row(lines[j])
            if not row or len(row) != len(header):
                break
            rows.append(row)
            j += 1

        out.append(
            {
                "headers": header,
                "rows": rows,
                "caption": _table_caption(lines, i),
                "start_line": i + 1,
                "end_line": j,
            }
        )
        i = j
    return out


def extract_markdown_figures(markdown: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for match in FIGURE_MD_RE.finditer(markdown or ""):
        out.append(
            {
                "alt_text": match.group(1).strip(),
                "image_ref": match.group(2).strip(),
            }
        )
    return out


def _resolve_figure_asset(doc_dir: Path, page_index: int, image_ref: str) -> Path | None:
    marker_root = doc_dir / "metadata" / "assets" / "structured" / "marker"
    if not marker_root.exists():
        return None

    ref = image_ref.split("#", 1)[0].split("?", 1)[0].strip()
    if not ref:
        return None
    ref_path = Path(ref)
    if ref_path.is_absolute() and ref_path.exists():
        return ref_path

    basename = ref_path.name
    if not basename:
        return None

    page_assets = marker_root / f"page_{page_index:04d}_assets"
    if page_assets.exists():
        exact = page_assets / ref
        if exact.exists():
            return exact
        matches = [p for p in page_assets.rglob(basename) if p.is_file()]
        if matches:
            return sorted(matches)[0]

    matches = [p for p in marker_root.rglob(basename) if p.is_file()]
    if matches:
        return sorted(matches)[0]
    return None


def _run_deplot_command(deplot_command: str, image_path: Path, timeout: int) -> dict[str, Any]:
    parts = shlex.split(deplot_command or "")
    if not parts:
        raise ValueError("Empty DePlot command.")

    has_placeholder = any("{image}" in part for part in parts)
    cmd = [part.replace("{image}", str(image_path)) for part in parts]
    if not has_placeholder:
        cmd.append(str(image_path))

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"DePlot command timed out after {timeout}s") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or f"DePlot command failed with code {proc.returncode}")

    stdout = (proc.stdout or "").strip()
    payload: dict[str, Any]
    if stdout:
        try:
            payload = json.loads(stdout)
            if not isinstance(payload, dict):
                payload = {"raw": payload}
        except json.JSONDecodeError:
            payload = {"raw_text": stdout}
    else:
        payload = {}
    return {"command": cmd, "output": payload}


def _write_table_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except Exception:
            return str(path)


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


def _stable_id(*parts: str) -> str:
    src = "|".join(parts)
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _bbox_from_polygons(polygons: list[list[list[float]]]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for poly in polygons:
        for point in poly:
            if len(point) < 2:
                continue
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except Exception:
                continue
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


TABLE_NUM_RE = re.compile(r"\b(?:table|tab\.?)\s*([0-9ivxlcdm]+)\b", re.IGNORECASE)
TABLE_CONTINUED_RE = re.compile(r"\b(cont(?:inued)?\.?)\b", re.IGNORECASE)


def _extract_table_number(caption: str) -> str:
    match = TABLE_NUM_RE.search(caption or "")
    if not match:
        return ""
    return match.group(1).strip().lower()


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_tr = False
        self.in_cell = False
        self.cell_tag = ""
        self.cell_buf: list[str] = []
        self.row_cells: list[tuple[str, str]] = []
        self.header_rows: list[list[str]] = []
        self.data_rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self.in_tr = True
            self.row_cells = []
            return
        if t in {"th", "td"} and self.in_tr:
            self.in_cell = True
            self.cell_tag = t
            self.cell_buf = []
            return
        if t == "br" and self.in_cell:
            self.cell_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"th", "td"} and self.in_cell:
            text = unescape("".join(self.cell_buf)).strip()
            self.row_cells.append((self.cell_tag, text))
            self.in_cell = False
            self.cell_tag = ""
            self.cell_buf = []
            return
        if t == "tr" and self.in_tr:
            if self.row_cells:
                tags = [k for k, _ in self.row_cells]
                vals = [v for _, v in self.row_cells]
                if all(k == "th" for k in tags):
                    self.header_rows.append(vals)
                else:
                    self.data_rows.append(vals)
            self.in_tr = False
            self.row_cells = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_buf.append(data)


def _parse_html_table_rows(table_html: str) -> tuple[list[list[str]], list[list[str]]]:
    parser = _HTMLTableParser()
    parser.feed(str(table_html or ""))
    parser.close()
    return parser.header_rows, parser.data_rows


def _normalize_fragment_from_marker(row: dict[str, Any], index: int) -> TableFragment:
    page = int(row.get("page") or 1)
    polygons = row.get("polygons")
    if not isinstance(polygons, list):
        polygons = []
    header_rows = row.get("header_rows")
    if not isinstance(header_rows, list):
        header_rows = []
    data_rows = row.get("data_rows")
    if not isinstance(data_rows, list):
        data_rows = []
    if not header_rows and not data_rows:
        html_table = str(row.get("html_table", "") or row.get("html", "") or "")
        if "<table" in html_table.lower():
            parsed_headers, parsed_rows = _parse_html_table_rows(html_table)
            header_rows = parsed_headers
            data_rows = parsed_rows
    caption_text = str(row.get("caption_text", "") or "")
    raw_group_id = str(row.get("table_group_id", "") or "").strip()
    table_group_id = raw_group_id or f"group_{page}_{index}"
    block_ids = row.get("table_block_ids")
    if not isinstance(block_ids, list):
        block_ids = []
    caption_block_id = str(row.get("caption_block_id", "") or "")
    quality = row.get("quality_metrics") if isinstance(row.get("quality_metrics"), dict) else {}
    frag_id = _stable_id(
        table_group_id,
        str(page),
        caption_text[:64],
        json.dumps(polygons, ensure_ascii=True, sort_keys=True),
    )
    return TableFragment(
        fragment_id=frag_id,
        table_group_id=table_group_id,
        explicit_group_id=bool(raw_group_id),
        table_block_ids=[str(x) for x in block_ids],
        caption_block_id=caption_block_id,
        note_block_ids=[str(x) for x in row.get("note_block_ids", []) if str(x)],
        page=page,
        polygons=polygons,
        bbox=_bbox_from_polygons(polygons),
        header_rows=[list(map(str, hr)) for hr in header_rows if isinstance(hr, list)],
        data_rows=[list(map(str, dr)) for dr in data_rows if isinstance(dr, list)],
        caption_text=caption_text,
        caption_confidence=float(row.get("caption_confidence", 0.9) or 0.9),
        source_format=str(row.get("source_format", "html") or "html"),
        quality_metrics=quality,
    )


def _normalized_header_tokens(frag: TableFragment) -> set[str]:
    if not frag.header_rows:
        return set()
    first = frag.header_rows[0]
    tokens: set[str] = set()
    for cell in first:
        for token in re.split(r"\W+", str(cell).lower()):
            cleaned = token.strip()
            if cleaned:
                tokens.add(cleaned)
    return tokens


def _header_similarity(lhs: TableFragment, rhs: TableFragment) -> float:
    l_tokens = _normalized_header_tokens(lhs)
    r_tokens = _normalized_header_tokens(rhs)
    if not l_tokens or not r_tokens:
        return 0.0
    overlap = l_tokens.intersection(r_tokens)
    union = l_tokens.union(r_tokens)
    return len(overlap) / max(len(union), 1)


def _can_merge_by_table_number(prev: TableFragment, current: TableFragment) -> bool:
    page_gap = current.page - prev.page
    if page_gap < 0 or page_gap > 1:
        return False
    prev_caption = str(prev.caption_text or "")
    current_caption = str(current.caption_text or "")
    if TABLE_CONTINUED_RE.search(prev_caption) or TABLE_CONTINUED_RE.search(current_caption):
        return True
    return _header_similarity(prev, current) >= 0.75


def _merge_table_fragments(fragments: list[TableFragment]) -> list[dict[str, Any]]:
    groups: dict[str, list[TableFragment]] = {}
    numbered_groups: dict[str, list[list[TableFragment]]] = {}
    for frag in sorted(fragments, key=lambda f: (f.page, f.fragment_id)):
        if frag.explicit_group_id:
            key = f"group:{frag.table_group_id}"
            groups.setdefault(key, []).append(frag)
            continue
        table_number = _extract_table_number(frag.caption_text)
        if not table_number:
            groups.setdefault(f"fragment:{frag.fragment_id}", []).append(frag)
            continue
        chains = numbered_groups.setdefault(table_number, [])
        if not chains:
            chains.append([frag])
            continue
        last_chain = chains[-1]
        if _can_merge_by_table_number(last_chain[-1], frag):
            last_chain.append(frag)
        else:
            chains.append([frag])

    for table_number, chains in numbered_groups.items():
        for idx, chain in enumerate(chains, start=1):
            groups[f"number:{table_number}:{idx}"] = chain

    out: list[dict[str, Any]] = []
    for key, frags in groups.items():
        frags_sorted = sorted(frags, key=lambda f: (f.page, f.fragment_id))
        header_rows = frags_sorted[0].header_rows
        merged_rows: list[list[str]] = []
        for frag in frags_sorted:
            for row in frag.data_rows:
                if merged_rows and row == merged_rows[-1]:
                    continue
                merged_rows.append(row)
        out.append(
            {
                "table_id": _stable_id(key, ",".join(f.fragment_id for f in frags_sorted)),
                "table_group_id": key,
                "fragment_ids": [f.fragment_id for f in frags_sorted],
                "pages": sorted({f.page for f in frags_sorted}),
                "caption_text": next((f.caption_text for f in frags_sorted if f.caption_text), ""),
                "header_rows": header_rows,
                "data_rows": merged_rows,
                "source_format": frags_sorted[0].source_format,
                "merge_confidence": 0.9 if len(frags_sorted) > 1 else 1.0,
            }
        )
    return out


def _quality_metrics(headers: list[str], rows: list[list[str]]) -> dict[str, Any]:
    total_cells = max(len(headers), 1) * max(len(rows), 1)
    flattened = [cell for row in rows for cell in row]
    empty_cells = sum(1 for cell in flattened if not str(cell).strip())
    repeated = 0
    if flattened:
        common = max((flattened.count(v) for v in set(flattened)))
        repeated = common
    row_lengths = [len(r) for r in rows] if rows else [len(headers)]
    unstable = sum(1 for n in row_lengths if n != len(headers))
    return {
        "empty_cell_ratio": empty_cells / max(total_cells, 1),
        "repeated_text_ratio": repeated / max(len(flattened), 1),
        "column_instability_ratio": unstable / max(len(row_lengths), 1),
    }


def _fails_quality(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("empty_cell_ratio", 0.0)) > 0.35
        or float(metrics.get("repeated_text_ratio", 0.0)) > 0.45
        or float(metrics.get("column_instability_ratio", 0.0)) > 0.20
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
    else:
        path.write_text("")


def _normalize_cell_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    return " ".join(text.split())


def _flatten_table_text(headers: list[str], rows: list[list[str]]) -> str:
    parts: list[str] = []
    if headers:
        parts.append(" | ".join(_normalize_cell_text(c) for c in headers))
    for row in rows:
        parts.append(" | ".join(_normalize_cell_text(c) for c in row))
    return "\n".join(parts)


def _extract_symbols(text: str) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for ch in SYMBOL_CHAR_RE.findall(str(text or "")):
        if ch in seen:
            continue
        seen.add(ch)
        ordered.append(ch)
    return "".join(ordered)


def _parse_ocr_html_table(path: Path) -> tuple[list[str], list[list[str]]]:
    raw = path.read_text()
    header_rows, data_rows = _parse_html_table_rows(raw)
    if not header_rows and not data_rows:
        markdown_tables = extract_markdown_tables(raw)
        if markdown_tables:
            first = markdown_tables[0]
            return list(first.get("headers", [])), [list(r) for r in first.get("rows", [])]
        return [], []
    headers = list(header_rows[0]) if header_rows else []
    rows = [list(r) for r in data_rows]
    return headers, rows


def _alnum_signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _cell_patch_decision(marker_value: str, ocr_value: str, *, allow_empty_fill: bool = True) -> tuple[str, str]:
    marker_text = _normalize_cell_text(marker_value)
    ocr_text = _normalize_cell_text(ocr_value)
    if not ocr_text:
        return marker_text, ""
    if allow_empty_fill and not marker_text:
        if _extract_symbols(ocr_text):
            return ocr_text, "empty_fill"
        return marker_text, ""
    if marker_text == ocr_text:
        return marker_text, ""

    marker_symbols = set(_extract_symbols(marker_text))
    ocr_symbols = set(_extract_symbols(ocr_text))
    if ocr_symbols.difference(marker_symbols):
        marker_sig = _alnum_signature(marker_text)
        ocr_sig = _alnum_signature(ocr_text)
        if marker_sig and ocr_sig:
            ratio = SequenceMatcher(None, marker_sig, ocr_sig).ratio()
            if ratio >= 0.50:
                return ocr_text, "symbol_patch"
        else:
            return ocr_text, "symbol_patch"

    if "<br" in str(marker_value or "") and "<br" not in str(ocr_value or ""):
        marker_sig = _alnum_signature(marker_text)
        ocr_sig = _alnum_signature(ocr_text)
        if marker_sig and marker_sig == ocr_sig:
            return ocr_text, "format_patch"

    return marker_text, ""


def _load_ocr_tables_by_page(ocr_root: Path) -> dict[int, list[dict[str, Any]]]:
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(ocr_root.glob("table_*_page_*.md")):
        match = OCR_TABLE_FILE_RE.search(path.name)
        if not match:
            continue
        page = int(match.group(2))
        ordinal = int(match.group(1))
        headers, rows = _parse_ocr_html_table(path)
        tables_by_page.setdefault(page, []).append(
            {
                "ordinal": ordinal,
                "headers": headers,
                "rows": rows,
                "path": path,
            }
        )
    for rows in tables_by_page.values():
        rows.sort(key=lambda row: int(row.get("ordinal", 0)))
    return tables_by_page


def _patch_table_grid(
    marker_headers: list[str],
    marker_rows: list[list[str]],
    ocr_headers: list[str],
    ocr_rows: list[list[str]],
) -> tuple[list[str], list[list[str]], int, dict[str, int]]:
    headers = list(marker_headers)
    rows = [list(row) for row in marker_rows]
    patch_count = 0
    patch_reasons: dict[str, int] = {}

    max_header_cols = min(len(headers), len(ocr_headers))
    for col in range(max_header_cols):
        merged, reason = _cell_patch_decision(headers[col], ocr_headers[col], allow_empty_fill=True)
        if reason:
            headers[col] = merged
            patch_count += 1
            patch_reasons[reason] = patch_reasons.get(reason, 0) + 1

    max_rows = min(len(rows), len(ocr_rows))
    for row_idx in range(max_rows):
        marker_row_sig = _alnum_signature(" ".join(str(cell) for cell in rows[row_idx]))
        ocr_row_sig = _alnum_signature(" ".join(str(cell) for cell in ocr_rows[row_idx]))
        row_similarity = SequenceMatcher(None, marker_row_sig, ocr_row_sig).ratio() if (marker_row_sig or ocr_row_sig) else 1.0
        allow_empty_fill = row_similarity >= 0.90
        max_cols = min(len(rows[row_idx]), len(ocr_rows[row_idx]))
        for col_idx in range(max_cols):
            merged, reason = _cell_patch_decision(
                rows[row_idx][col_idx],
                ocr_rows[row_idx][col_idx],
                allow_empty_fill=allow_empty_fill,
            )
            if reason:
                rows[row_idx][col_idx] = merged
                patch_count += 1
                patch_reasons[reason] = patch_reasons.get(reason, 0) + 1
    return headers, rows, patch_count, patch_reasons


def _merge_marker_tables_with_ocr_html(
    *,
    canonical_tables: list[dict[str, Any]],
    doc_dir: Path,
    ocr_html_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    report_path = qa_root / "table_ocr_merge.json"

    if not canonical_tables:
        payload = {
            "enabled": True,
            "tables_considered": 0,
            "tables_matched": 0,
            "tables_patched": 0,
            "cells_patched": 0,
            "report_path": _portable_path(report_path, doc_dir),
            "results": [],
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        return canonical_tables, payload

    ocr_root = ocr_html_dir or (qa_root / "bbox_ocr_outputs")
    if not ocr_root.exists():
        payload = {
            "enabled": True,
            "tables_considered": len(canonical_tables),
            "tables_matched": 0,
            "tables_patched": 0,
            "cells_patched": 0,
            "report_path": _portable_path(report_path, doc_dir),
            "results": [],
            "warning": f"ocr_html_dir_missing:{ocr_root}",
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        return canonical_tables, payload

    ocr_by_page = _load_ocr_tables_by_page(ocr_root)
    marker_by_page: dict[int, list[int]] = {}
    for idx, table in enumerate(canonical_tables):
        pages = [int(p) for p in table.get("pages", []) if int(p) > 0]
        primary_page = min(pages) if pages else 1
        marker_by_page.setdefault(primary_page, []).append(idx)
    for page in marker_by_page:
        marker_by_page[page].sort(key=lambda i: str(canonical_tables[i].get("table_id", "")))

    merged_tables: list[dict[str, Any]] = [dict(table) for table in canonical_tables]
    results: list[dict[str, Any]] = []
    tables_matched = 0
    tables_patched = 0
    cells_patched = 0

    for page in sorted(marker_by_page):
        marker_indices = marker_by_page.get(page, [])
        ocr_tables = ocr_by_page.get(page, [])
        for local_idx, marker_table_index in enumerate(marker_indices):
            table = merged_tables[marker_table_index]
            ocr_table = ocr_tables[local_idx] if local_idx < len(ocr_tables) else None
            table_id = str(table.get("table_id", ""))
            header_rows = table.get("header_rows", [])
            marker_headers = list(header_rows[0]) if isinstance(header_rows, list) and header_rows else []
            marker_rows = [list(r) for r in table.get("data_rows", []) if isinstance(r, list)]
            marker_symbols_before = _extract_symbols(_flatten_table_text(marker_headers, marker_rows))
            table_cell_patches = 0
            patch_reasons: dict[str, int] = {}

            if ocr_table is None:
                results.append(
                    {
                        "table_id": table_id,
                        "page": page,
                        "status": "missing_ocr_html",
                        "cells_patched": 0,
                        "ocr_html_path": "",
                    }
                )
                continue

            tables_matched += 1
            ocr_headers = [str(x) for x in ocr_table.get("headers", [])]
            ocr_rows = [[str(x) for x in row] for row in ocr_table.get("rows", []) if isinstance(row, list)]
            marker_headers, marker_rows, table_cell_patches, patch_reasons = _patch_table_grid(
                marker_headers=marker_headers,
                marker_rows=marker_rows,
                ocr_headers=ocr_headers,
                ocr_rows=ocr_rows,
            )

            table["header_rows"] = [marker_headers] if marker_headers else []
            table["data_rows"] = marker_rows
            if table_cell_patches:
                table["source_format"] = "hybrid"
                table["ocr_merge"] = {
                    "enabled": True,
                    "ocr_html_path": _portable_path(Path(ocr_table["path"]), doc_dir),
                    "cells_patched": table_cell_patches,
                    "patch_reasons": patch_reasons,
                }
                tables_patched += 1
                cells_patched += table_cell_patches
                status = "patched"
            else:
                table["ocr_merge"] = {
                    "enabled": True,
                    "ocr_html_path": _portable_path(Path(ocr_table["path"]), doc_dir),
                    "cells_patched": 0,
                    "patch_reasons": {},
                }
                status = "unchanged"

            results.append(
                {
                    "table_id": table_id,
                    "page": page,
                    "status": status,
                    "cells_patched": table_cell_patches,
                    "patch_reasons": patch_reasons,
                    "ocr_html_path": _portable_path(Path(ocr_table["path"]), doc_dir),
                    "marker_symbols_before": marker_symbols_before,
                    "marker_symbols_after": _extract_symbols(_flatten_table_text(marker_headers, marker_rows)),
                    "ocr_symbols": _extract_symbols(_flatten_table_text(ocr_headers, ocr_rows)),
                }
            )

    payload = {
        "enabled": True,
        "tables_considered": len(canonical_tables),
        "tables_matched": tables_matched,
        "tables_patched": tables_patched,
        "cells_patched": cells_patched,
        "report_path": _portable_path(report_path, doc_dir),
        "results": results,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return merged_tables, payload


def compare_marker_tables_with_ocr_html(
    *,
    doc_dir: Path,
    ocr_html_dir: Path | None = None,
) -> dict[str, Any]:
    qa_dir = doc_dir / "metadata" / "assets" / "structured" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / "table_ocr_html_compare.json"
    tables_manifest = doc_dir / "metadata" / "assets" / "structured" / "extracted" / "tables" / "manifest.jsonl"
    if not tables_manifest.exists():
        payload = {
            "doc_dir": str(doc_dir),
            "tables_compared": 0,
            "avg_similarity": 0.0,
            "results": [],
            "error": "missing tables manifest",
            "report_path": _portable_path(report_path, doc_dir),
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        return payload

    html_root = ocr_html_dir or (qa_dir / "bbox_ocr_outputs")
    if not html_root.exists():
        payload = {
            "doc_dir": str(doc_dir),
            "ocr_html_dir": str(html_root),
            "tables_compared": 0,
            "avg_similarity": 0.0,
            "results": [],
            "error": "ocr html directory missing",
            "report_path": _portable_path(report_path, doc_dir),
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        return payload

    table_rows = _load_jsonl(tables_manifest)
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in table_rows:
        page = int(row.get("page") or 0)
        if page <= 0:
            continue
        tables_by_page.setdefault(page, []).append(row)
    for rows in tables_by_page.values():
        rows.sort(key=lambda r: str(r.get("table_id", "")))

    ocr_files_by_page: dict[int, list[Path]] = {}
    for path in sorted(html_root.glob("table_*_page_*.md")):
        match = OCR_TABLE_FILE_RE.search(path.name)
        if not match:
            continue
        page = int(match.group(2))
        ocr_files_by_page.setdefault(page, []).append(path)

    results: list[dict[str, Any]] = []
    similarities: list[float] = []
    for page in sorted(tables_by_page):
        marker_tables = tables_by_page.get(page, [])
        ocr_tables = ocr_files_by_page.get(page, [])
        for idx, marker_row in enumerate(marker_tables):
            ocr_path = ocr_tables[idx] if idx < len(ocr_tables) else None
            marker_headers = [str(x) for x in marker_row.get("headers", [])]
            marker_data_rows = [[str(x) for x in row] for row in marker_row.get("rows", []) if isinstance(row, list)]
            marker_text = _flatten_table_text(marker_headers, marker_data_rows)

            if ocr_path is None:
                results.append(
                    {
                        "table_id": str(marker_row.get("table_id", "")),
                        "page": page,
                        "ocr_html_path": "",
                        "similarity": 0.0,
                        "marker_symbols": _extract_symbols(marker_text),
                        "ocr_symbols": "",
                        "missing_in_marker_vs_ocr": "",
                        "missing_in_ocr_vs_marker": _extract_symbols(marker_text),
                        "status": "missing_ocr_html",
                    }
                )
                continue

            ocr_headers, ocr_rows = _parse_ocr_html_table(ocr_path)
            ocr_text = _flatten_table_text(ocr_headers, ocr_rows)
            similarity = SequenceMatcher(None, marker_text, ocr_text).ratio() if (marker_text or ocr_text) else 1.0
            similarities.append(float(similarity))

            marker_symbols = _extract_symbols(marker_text)
            ocr_symbols = _extract_symbols(ocr_text)
            missing_in_marker = "".join(ch for ch in ocr_symbols if ch not in marker_symbols)
            missing_in_ocr = "".join(ch for ch in marker_symbols if ch not in ocr_symbols)

            results.append(
                {
                    "table_id": str(marker_row.get("table_id", "")),
                    "page": page,
                    "ocr_html_path": _portable_path(ocr_path, doc_dir),
                    "marker_rows": len(marker_data_rows),
                    "marker_cols": len(marker_headers),
                    "ocr_rows": len(ocr_rows),
                    "ocr_cols": len(ocr_headers),
                    "similarity": round(float(similarity), 4),
                    "marker_symbols": marker_symbols,
                    "ocr_symbols": ocr_symbols,
                    "missing_in_marker_vs_ocr": missing_in_marker,
                    "missing_in_ocr_vs_marker": missing_in_ocr,
                    "status": "ok",
                }
            )

    avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0
    payload = {
        "doc_dir": str(doc_dir),
        "ocr_html_dir": str(html_root),
        "tables_compared": len(similarities),
        "avg_similarity": avg_similarity,
        "results": results,
        "report_path": _portable_path(report_path, doc_dir),
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return payload


def _reconcile_with_grobid(
    *,
    canonical_tables: list[dict[str, Any]],
    grobid_tables: list[dict[str, Any]],
) -> list[QAFlag]:
    flags: list[QAFlag] = []
    marker_pages = sorted({int(p) for t in canonical_tables for p in t.get("pages", [])})
    grobid_pages = sorted({int(t.get("page") or 0) for t in grobid_tables if int(t.get("page") or 0) > 0})

    if len(canonical_tables) != len(grobid_tables):
        flags.append(
            QAFlag(
                flag_id=_stable_id("qa", "count", str(len(canonical_tables)), str(len(grobid_tables))),
                severity="warn",
                type="count_mismatch",
                page=0,
                table_ref="*",
                details=f"marker={len(canonical_tables)} grobid={len(grobid_tables)}",
            )
        )
    if marker_pages != grobid_pages:
        flags.append(
            QAFlag(
                flag_id=_stable_id("qa", "pages", ",".join(map(str, marker_pages)), ",".join(map(str, grobid_pages))),
                severity="warn",
                type="page_mismatch",
                page=0,
                table_ref="*",
                details=f"marker_pages={marker_pages} grobid_pages={grobid_pages}",
            )
        )
    marker_nums = sorted({_extract_table_number(str(t.get("caption_text", ""))) for t in canonical_tables if _extract_table_number(str(t.get("caption_text", "")))})
    grobid_nums = sorted({_extract_table_number(str(t.get("label", "")) + " " + str(t.get("caption_text", ""))) for t in grobid_tables if _extract_table_number(str(t.get("label", "")) + " " + str(t.get("caption_text", "")))})
    if marker_nums and grobid_nums and marker_nums != grobid_nums:
        flags.append(
            QAFlag(
                flag_id=_stable_id("qa", "caption-num", ",".join(marker_nums), ",".join(grobid_nums)),
                severity="warn",
                type="caption_number_mismatch",
                page=0,
                table_ref="*",
                details=f"marker_numbers={marker_nums} grobid_numbers={grobid_nums}",
            )
        )
    return flags


def build_structured_exports(
    doc_dir: Path,
    *,
    deplot_command: str = "",
    deplot_timeout: int = 90,
    table_source: str = "marker-first",
    table_ocr_merge: bool = True,
    ocr_html_dir: Path | None = None,
    table_quality_gate: bool = True,
    table_escalation: str = "auto",
    table_escalation_max: int = 20,
    table_qa_mode: str = "warn",
    grobid_status: str = "ok",
) -> StructuredExportSummary:
    summary = StructuredExportSummary()
    pages_dir = doc_dir / "pages"
    if not pages_dir.exists():
        return summary

    extracted_root = doc_dir / "metadata" / "assets" / "structured" / "extracted"
    tables_dir = extracted_root / "tables"
    figures_dir = extracted_root / "figures"
    deplot_dir = figures_dir / "deplot"
    if tables_dir.exists():
        shutil.rmtree(tables_dir)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    fragments: list[TableFragment] = []
    canonical_tables: list[dict[str, Any]] = []
    qa_flags: list[QAFlag] = []
    markdown_merge_results: list[dict[str, Any]] = []
    markdown_tables_matched = 0
    markdown_tables_patched = 0
    markdown_cells_patched = 0
    ocr_tables_by_page: dict[int, list[dict[str, Any]]] = {}
    ocr_root = (ocr_html_dir or (doc_dir / "metadata" / "assets" / "structured" / "qa" / "bbox_ocr_outputs"))
    if table_ocr_merge and ocr_root.exists():
        ocr_tables_by_page = _load_ocr_tables_by_page(ocr_root)

    marker_tables_raw = doc_dir / "metadata" / "assets" / "structured" / "marker" / "tables_raw.jsonl"
    if table_source == "marker-first" and marker_tables_raw.exists():
        marker_rows = _load_jsonl(marker_tables_raw)
        for idx, row in enumerate(marker_rows, start=1):
            fragments.append(_normalize_fragment_from_marker(row, idx))
        canonical_tables = _merge_table_fragments(fragments)
        if table_ocr_merge:
            canonical_tables, merge_summary = _merge_marker_tables_with_ocr_html(
                canonical_tables=canonical_tables,
                doc_dir=doc_dir,
                ocr_html_dir=ocr_html_dir,
            )
            summary.ocr_merge = merge_summary

    for page_path in sorted(pages_dir.glob("*.md")):
        if not page_path.stem.isdigit():
            continue
        page_index = int(page_path.stem)
        markdown = page_path.read_text()

        tables = [] if canonical_tables else extract_markdown_tables(markdown)
        for t_index, table in enumerate(tables, start=1):
            headers = list(table["headers"])
            rows = [list(r) for r in table["rows"]]
            if table_ocr_merge:
                ocr_tables = ocr_tables_by_page.get(page_index, [])
                ocr_table = ocr_tables[t_index - 1] if t_index - 1 < len(ocr_tables) else None
                if ocr_table is not None:
                    markdown_tables_matched += 1
                    marker_symbols_before = _extract_symbols(_flatten_table_text(headers, rows))
                    ocr_headers = [str(x) for x in ocr_table.get("headers", [])]
                    ocr_rows = [[str(x) for x in row] for row in ocr_table.get("rows", []) if isinstance(row, list)]
                    headers, rows, patched_cells, patch_reasons = _patch_table_grid(
                        marker_headers=headers,
                        marker_rows=rows,
                        ocr_headers=ocr_headers,
                        ocr_rows=ocr_rows,
                    )
                    if patched_cells:
                        markdown_tables_patched += 1
                        markdown_cells_patched += patched_cells
                        merge_status = "patched"
                    else:
                        merge_status = "unchanged"
                    markdown_merge_results.append(
                        {
                            "table_id": f"p{page_index:04d}_t{t_index:02d}",
                            "page": page_index,
                            "status": merge_status,
                            "cells_patched": patched_cells,
                            "patch_reasons": patch_reasons,
                            "ocr_html_path": _portable_path(Path(ocr_table["path"]), doc_dir),
                            "marker_symbols_before": marker_symbols_before,
                            "marker_symbols_after": _extract_symbols(_flatten_table_text(headers, rows)),
                            "ocr_symbols": _extract_symbols(_flatten_table_text(ocr_headers, ocr_rows)),
                        }
                    )
                else:
                    markdown_merge_results.append(
                        {
                            "table_id": f"p{page_index:04d}_t{t_index:02d}",
                            "page": page_index,
                            "status": "missing_ocr_html",
                            "cells_patched": 0,
                            "ocr_html_path": "",
                        }
                    )
            table_id = f"p{page_index:04d}_t{t_index:02d}"
            csv_path = tables_dir / f"{table_id}.csv"
            json_path = tables_dir / f"{table_id}.json"
            _write_table_csv(csv_path, headers, rows)
            payload = {
                "table_id": table_id,
                "page": page_index,
                "caption": table["caption"],
                "headers": headers,
                "rows": rows,
                "csv_path": _portable_path(csv_path, doc_dir),
            }
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
            table_rows.append(payload)
            summary.table_count += 1

        figures = extract_markdown_figures(markdown)
        for f_index, figure in enumerate(figures, start=1):
            figure_id = f"p{page_index:04d}_f{f_index:02d}"
            resolved = _resolve_figure_asset(doc_dir, page_index, figure["image_ref"])
            entry: dict[str, Any] = {
                "figure_id": figure_id,
                "page": page_index,
                "alt_text": figure["alt_text"],
                "image_ref": figure["image_ref"],
                "resolved_path": _portable_path(resolved, doc_dir) if resolved else "",
                "deplot_path": "",
                "deplot_error": "",
            }
            if resolved is None:
                summary.unresolved_figure_count += 1
            elif deplot_command.strip():
                out_path = deplot_dir / f"{figure_id}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    result = _run_deplot_command(deplot_command, resolved, deplot_timeout)
                    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True))
                    entry["deplot_path"] = _portable_path(out_path, doc_dir)
                    summary.deplot_count += 1
                except Exception as exc:  # noqa: BLE001
                    message = f"{figure_id}: {exc}"
                    entry["deplot_error"] = str(exc)
                    summary.errors.append(message)
            figure_rows.append(entry)
            summary.figure_count += 1

    if canonical_tables:
        fragment_rows = []
        for frag in fragments:
            fragment_rows.append(
                {
                    "fragment_id": frag.fragment_id,
                    "table_group_id": frag.table_group_id,
                    "table_block_ids": frag.table_block_ids,
                    "caption_block_id": frag.caption_block_id,
                    "note_block_ids": frag.note_block_ids,
                    "page": frag.page,
                    "polygons": frag.polygons,
                    "bbox": frag.bbox,
                    "header_rows": frag.header_rows,
                    "data_rows": frag.data_rows,
                    "caption_text": frag.caption_text,
                    "caption_confidence": frag.caption_confidence,
                    "source_format": frag.source_format,
                    "quality_metrics": frag.quality_metrics,
                }
            )
        _write_jsonl(extracted_root / "table_fragments.jsonl", fragment_rows)
        canonical_rows: list[dict[str, Any]] = []
        for table in canonical_tables:
            header = table.get("header_rows", [])
            headers = list(header[0]) if header else []
            rows = [list(r) for r in table.get("data_rows", []) if isinstance(r, list)]
            metrics = _quality_metrics(headers, rows)
            if table_quality_gate and _fails_quality(metrics):
                msg = f"{table.get('table_id')}: quality gate failed"
                summary.errors.append(msg)
                if table_escalation in {"auto", "always"} and table_escalation_max > 0:
                    table_escalation_max -= 1
            table["quality_metrics"] = metrics
            canonical_rows.append(table)
            csv_path = tables_dir / f"{table['table_id']}.csv"
            _write_table_csv(csv_path, headers, rows)
            json_path = tables_dir / f"{table['table_id']}.json"
            json_payload = dict(table)
            json_payload["csv_path"] = _portable_path(csv_path, doc_dir)
            json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=True))
            table_rows.append(
                {
                    "table_id": table["table_id"],
                    "page": min(table.get("pages", [1])),
                    "caption": table.get("caption_text", ""),
                    "headers": headers,
                    "rows": rows,
                    "csv_path": _portable_path(csv_path, doc_dir),
                }
            )
            summary.table_count += 1
        _write_jsonl(tables_dir / "canonical.jsonl", canonical_rows)

        if table_qa_mode != "off":
            grobid_tables_path = doc_dir / "metadata" / "assets" / "structured" / "grobid" / "figures_tables.jsonl"
            grobid_tables = [row for row in _load_jsonl(grobid_tables_path) if str(row.get("type", "")).lower() == "table"]
            if grobid_status != "ok":
                qa_flags.append(
                    QAFlag(
                        flag_id=_stable_id("qa-skip", grobid_status),
                        severity="info",
                        type="qa_skipped",
                        page=0,
                        table_ref="*",
                        details=f"grobid_status={grobid_status}",
                    )
                )
            else:
                qa_flags.extend(
                    _reconcile_with_grobid(
                        canonical_tables=canonical_tables,
                        grobid_tables=grobid_tables,
                    )
                )
        qa_rows = [
            {
                "flag_id": flag.flag_id,
                "severity": flag.severity,
                "type": flag.type,
                "page": flag.page,
                "table_ref": flag.table_ref,
                "details": flag.details,
            }
            for flag in qa_flags
        ]
        qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
        _write_jsonl(qa_root / "table_flags.jsonl", qa_rows)
        (qa_root / "table_reconciliation.json").write_text(
            json.dumps(
                {
                    "qa_mode": table_qa_mode,
                    "marker_table_count": len(canonical_tables),
                    "flag_count": len(qa_rows),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        if table_qa_mode == "strict" and any(flag.severity in {"warn", "error"} for flag in qa_flags):
            raise RuntimeError("Table QA strict mode failed due to disagreements.")
    elif table_ocr_merge:
        qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
        qa_root.mkdir(parents=True, exist_ok=True)
        merge_payload = {
            "enabled": True,
            "tables_considered": len(table_rows),
            "tables_matched": markdown_tables_matched,
            "tables_patched": markdown_tables_patched,
            "cells_patched": markdown_cells_patched,
            "report_path": _portable_path(qa_root / "table_ocr_merge.json", doc_dir),
            "results": markdown_merge_results,
            "mode": "markdown_tables",
        }
        (qa_root / "table_ocr_merge.json").write_text(json.dumps(merge_payload, indent=2, ensure_ascii=True))
        summary.ocr_merge = merge_payload

    tables_jsonl = tables_dir / "manifest.jsonl"
    if table_rows:
        tables_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in table_rows) + "\n")
    else:
        tables_jsonl.write_text("")

    figures_jsonl = figures_dir / "manifest.jsonl"
    if figure_rows:
        figures_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in figure_rows) + "\n")
    else:
        figures_jsonl.write_text("")

    manifest_payload = {
        "table_count": summary.table_count,
        "figure_count": summary.figure_count,
        "deplot_count": summary.deplot_count,
        "unresolved_figure_count": summary.unresolved_figure_count,
        "errors": summary.errors,
        "tables_manifest": _portable_path(tables_jsonl, doc_dir),
        "figures_manifest": _portable_path(figures_jsonl, doc_dir),
    }
    extracted_root.mkdir(parents=True, exist_ok=True)
    (extracted_root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=True))
    return summary
