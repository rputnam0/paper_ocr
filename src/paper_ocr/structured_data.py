from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FIGURE_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@dataclass
class StructuredExportSummary:
    table_count: int = 0
    figure_count: int = 0
    deplot_count: int = 0
    unresolved_figure_count: int = 0
    errors: list[str] = field(default_factory=list)


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


def build_structured_exports(
    doc_dir: Path,
    *,
    deplot_command: str = "",
    deplot_timeout: int = 90,
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

    for page_path in sorted(pages_dir.glob("*.md")):
        if not page_path.stem.isdigit():
            continue
        page_index = int(page_path.stem)
        markdown = page_path.read_text()

        tables = extract_markdown_tables(markdown)
        for t_index, table in enumerate(tables, start=1):
            table_id = f"p{page_index:04d}_t{t_index:02d}"
            csv_path = tables_dir / f"{table_id}.csv"
            json_path = tables_dir / f"{table_id}.json"
            _write_table_csv(csv_path, table["headers"], table["rows"])
            payload = {
                "table_id": table_id,
                "page": page_index,
                "caption": table["caption"],
                "headers": table["headers"],
                "rows": table["rows"],
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
