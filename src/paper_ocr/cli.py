from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz
from PIL import Image
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .anchoring import build_anchored_prompt, build_unanchored_prompt, extract_anchors
from .bibliography import (
    bibliography_prompt,
    citation_from_bibliography,
    extract_json_object,
    normalize_bibliography,
)
from .client import call_olmocr, call_text_model
from .data_audit import format_audit_report, run_data_audit
from .discoverability import (
    abstract_extraction_prompt,
    discoverability_aggregate_prompt,
    discoverability_chunk_prompt,
    first_pages_excerpt,
    is_useful_discovery,
    normalize_discovery,
    render_group_readme,
    split_markdown_for_discovery,
)
from .facts import export_facts_for_doc
from .ingest import discover_pdfs, doc_id_from_sha, file_sha256, output_group_name
from .inspect import compute_text_heuristics, decide_route, is_text_only_candidate
from .postprocess import parse_yaml_front_matter
from .render import render_page
from .schemas import new_manifest
from .store import ensure_dirs, write_json, write_text
from .structured_data import build_structured_exports, compare_marker_tables_with_ocr_html
from .structured_extract import (
    build_render_contract,
    grobid_coords_to_px,
    is_structured_candidate_doc,
    normalize_markdown_for_llm,
    run_grobid_doc,
    run_marker_doc,
    run_marker_page,
)
from .table_eval import evaluate_table_pipeline
from .telegram_fetch import FetchTelegramConfig, fetch_from_telegram

METADATA_MODEL_DEFAULT = "nvidia/Nemotron-3-Nano-30B-A3B"
MIN_DELAY_DEFAULT = "4"
MAX_DELAY_DEFAULT = "8"
RESPONSE_TIMEOUT_DEFAULT = 15
SEARCH_TIMEOUT_DEFAULT = 40
CSV_JOB_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
UNDERSCORE_RUN_RE = re.compile(r"_+")
DIGITAL_STRUCTURED_DEFAULT = "auto"
MARKER_COMMAND_DEFAULT = "marker_single"
MARKER_URL_DEFAULT = ""
MARKER_TIMEOUT_DEFAULT = "120"
GROBID_TIMEOUT_DEFAULT = "60"
DEPLOT_TIMEOUT_DEFAULT = "90"
TABLE_HEADER_OCR_MODEL_DEFAULT = "allenai/olmOCR-2-7B-1025"
TABLE_HEADER_OCR_MAX_TOKENS_DEFAULT = 1400
DEFAULT_FETCH_OUTPUT_ROOT = Path("data/jobs")
LEGACY_FETCH_OUTPUT_ROOT = Path("data/telegram_jobs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="paper-ocr")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run OCR on PDFs")
    run.add_argument("in_dir", type=Path)
    run.add_argument("out_dir", type=Path)
    run.add_argument("--workers", type=int, default=32)
    run.add_argument("--model", type=str, default="allenai/olmOCR-2-7B-1025")
    run.add_argument("--base-url", type=str, default="https://api.deepinfra.com/v1/openai")
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--force", action="store_true")
    run.add_argument("--mode", choices=["auto", "anchored", "unanchored"], default="auto")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--scan-preprocess", action="store_true")
    run.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable text-only extraction for high-quality text layers (default: enabled)",
    )
    run.add_argument("--metadata-model", type=str, default=METADATA_MODEL_DEFAULT)
    run.add_argument(
        "--digital-structured",
        choices=["off", "auto", "on"],
        default=os.getenv("PAPER_OCR_DIGITAL_STRUCTURED", DIGITAL_STRUCTURED_DEFAULT),
    )
    run.add_argument(
        "--structured-backend",
        choices=["marker", "hybrid"],
        default="hybrid",
    )
    run.add_argument(
        "--marker-command",
        type=str,
        default=os.getenv("PAPER_OCR_MARKER_COMMAND", MARKER_COMMAND_DEFAULT),
    )
    run.add_argument(
        "--marker-url",
        type=str,
        default=os.getenv("PAPER_OCR_MARKER_URL", MARKER_URL_DEFAULT),
        help="Optional Marker service base URL (for example http://127.0.0.1:8008).",
    )
    run.add_argument(
        "--marker-timeout",
        type=int,
        default=int(os.getenv("PAPER_OCR_MARKER_TIMEOUT", MARKER_TIMEOUT_DEFAULT)),
    )
    run.add_argument(
        "--grobid-url",
        type=str,
        default=os.getenv("PAPER_OCR_GROBID_URL", ""),
    )
    run.add_argument(
        "--grobid-timeout",
        type=int,
        default=int(os.getenv("PAPER_OCR_GROBID_TIMEOUT", GROBID_TIMEOUT_DEFAULT)),
    )
    run.add_argument("--structured-max-workers", type=int, default=4)
    run.add_argument(
        "--structured-asset-level",
        choices=["standard", "full"],
        default="standard",
    )
    run.add_argument(
        "--extract-structured-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export machine-readable tables/figures from page markdown and Marker assets (default: enabled).",
    )
    run.add_argument(
        "--deplot-command",
        type=str,
        default=os.getenv("PAPER_OCR_DEPLOT_COMMAND", ""),
        help="Optional external command for chart-to-table extraction; supports {image} placeholder.",
    )
    run.add_argument(
        "--deplot-timeout",
        type=int,
        default=int(os.getenv("PAPER_OCR_DEPLOT_TIMEOUT", DEPLOT_TIMEOUT_DEFAULT)),
    )
    run.add_argument(
        "--marker-localize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Marker document-level localization artifacts for all PDFs (default: enabled).",
    )
    run.add_argument(
        "--marker-localize-profile",
        choices=["localize_only", "full_json"],
        default="full_json",
    )
    run.add_argument(
        "--layout-fallback",
        choices=["none", "surya"],
        default="surya",
        help="Fallback layout detector for pages where Marker localization lacks geometry.",
    )
    run.add_argument(
        "--table-source",
        choices=["marker-first", "markdown-only"],
        default="marker-first",
    )
    run.add_argument(
        "--table-ocr-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically merge OCR HTML table cells into Marker tables to recover symbols/formatting.",
    )
    run.add_argument(
        "--table-ocr-merge-scope",
        choices=["header", "full"],
        default="header",
        help="Merge OCR data into table headers only (default) or full table grid.",
    )
    run.add_argument(
        "--table-header-ocr-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate missing table header OCR artifacts before merge.",
    )
    run.add_argument(
        "--table-header-ocr-model",
        type=str,
        default=os.getenv("PAPER_OCR_TABLE_HEADER_OCR_MODEL", TABLE_HEADER_OCR_MODEL_DEFAULT),
    )
    run.add_argument(
        "--table-header-ocr-max-tokens",
        type=int,
        default=int(os.getenv("PAPER_OCR_TABLE_HEADER_OCR_MAX_TOKENS", str(TABLE_HEADER_OCR_MAX_TOKENS_DEFAULT))),
    )
    run.add_argument(
        "--table-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--table-escalation",
        choices=["off", "auto", "always"],
        default="auto",
    )
    run.add_argument("--table-escalation-max", type=int, default=20)
    run.add_argument(
        "--table-qa-mode",
        choices=["off", "warn", "strict"],
        default="warn",
    )
    run.add_argument(
        "--table-artifact-mode",
        choices=["permissive", "strict"],
        default="permissive",
        help="Artifact gating policy for large runs; strict fails documents when required table artifacts are missing.",
    )
    run.add_argument(
        "--compare-ocr-html",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare Marker-extracted tables against OCR HTML table outputs in structured QA artifacts.",
    )
    run.add_argument(
        "--ocr-html-dir",
        type=Path,
        default=None,
        help="Optional directory containing OCR HTML tables (default: metadata/assets/structured/qa/bbox_ocr_outputs).",
    )

    fetch = sub.add_parser("fetch-telegram", help="Fetch PDFs from Telegram bot using DOI CSV")
    fetch.add_argument("doi_csv", type=Path)
    fetch.add_argument("output_root", type=Path, nargs="?", default=Path("data/jobs"))
    fetch.add_argument("--doi-column", type=str, default="DOI")
    fetch.add_argument("--target-bot", type=str, default=os.getenv("TARGET_BOT", ""))
    fetch.add_argument("--session-name", type=str, default="nexus_session")
    fetch.add_argument("--min-delay", type=float, default=float(os.getenv("MIN_DELAY", MIN_DELAY_DEFAULT)))
    fetch.add_argument("--max-delay", type=float, default=float(os.getenv("MAX_DELAY", MAX_DELAY_DEFAULT)))
    fetch.add_argument("--response-timeout", type=int, default=RESPONSE_TIMEOUT_DEFAULT)
    fetch.add_argument("--search-timeout", type=int, default=SEARCH_TIMEOUT_DEFAULT)
    fetch.add_argument("--debug", action="store_true")
    fetch.add_argument("--report-file", type=Path, default=None)
    fetch.add_argument("--failed-file", type=Path, default=None)

    export = sub.add_parser(
        "export-structured-data",
        help="Process existing OCR output folders into machine-readable table/figure artifacts",
    )
    export.add_argument("ocr_out_dir", type=Path)
    export.add_argument(
        "--deplot-command",
        type=str,
        default=os.getenv("PAPER_OCR_DEPLOT_COMMAND", ""),
        help="Optional external command for chart-to-table extraction; supports {image} placeholder.",
    )
    export.add_argument(
        "--deplot-timeout",
        type=int,
        default=int(os.getenv("PAPER_OCR_DEPLOT_TIMEOUT", DEPLOT_TIMEOUT_DEFAULT)),
    )
    export.add_argument(
        "--table-source",
        choices=["marker-first", "markdown-only"],
        default="marker-first",
    )
    export.add_argument(
        "--table-ocr-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically merge OCR HTML table cells into Marker tables to recover symbols/formatting.",
    )
    export.add_argument(
        "--table-ocr-merge-scope",
        choices=["header", "full"],
        default="header",
        help="Merge OCR data into table headers only (default) or full table grid.",
    )
    export.add_argument(
        "--table-header-ocr-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate missing table header OCR artifacts before merge.",
    )
    export.add_argument(
        "--table-header-ocr-model",
        type=str,
        default=os.getenv("PAPER_OCR_TABLE_HEADER_OCR_MODEL", TABLE_HEADER_OCR_MODEL_DEFAULT),
    )
    export.add_argument(
        "--table-header-ocr-max-tokens",
        type=int,
        default=int(os.getenv("PAPER_OCR_TABLE_HEADER_OCR_MAX_TOKENS", str(TABLE_HEADER_OCR_MAX_TOKENS_DEFAULT))),
    )
    export.add_argument(
        "--table-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    export.add_argument(
        "--table-escalation",
        choices=["off", "auto", "always"],
        default="auto",
    )
    export.add_argument("--table-escalation-max", type=int, default=20)
    export.add_argument(
        "--table-qa-mode",
        choices=["off", "warn", "strict"],
        default="warn",
    )
    export.add_argument(
        "--table-artifact-mode",
        choices=["permissive", "strict"],
        default="permissive",
        help="Artifact gating policy for large runs; strict fails documents when required table artifacts are missing.",
    )
    export.add_argument(
        "--compare-ocr-html",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare Marker-extracted tables against OCR HTML table outputs in structured QA artifacts.",
    )
    export.add_argument(
        "--ocr-html-dir",
        type=Path,
        default=None,
        help="Optional directory containing OCR HTML tables (default: metadata/assets/structured/qa/bbox_ocr_outputs).",
    )

    export_facts = sub.add_parser(
        "export-facts",
        help="Export normalized property records from structured table/figure artifacts.",
    )
    export_facts.add_argument("ocr_out_dir", type=Path)

    eval_tables = sub.add_parser(
        "eval-table-pipeline",
        help="Evaluate table extraction quality against a gold set.",
    )
    eval_tables.add_argument("gold_dir", type=Path)
    eval_tables.add_argument("pred_dir", type=Path)
    eval_tables.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline metrics JSON for regression checks.",
    )
    eval_tables.add_argument(
        "--strict-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when regression thresholds are exceeded against baseline metrics.",
    )
    eval_tables.add_argument("--max-precision-drop", type=float, default=0.03)
    eval_tables.add_argument("--max-recall-drop", type=float, default=0.03)
    eval_tables.add_argument("--min-numeric-parse", type=float, default=0.8)

    audit = sub.add_parser("data-audit", help="Validate data/ folder organization contract")
    audit.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    audit.add_argument("--strict", action="store_true", help="Exit non-zero on contract violations.")
    audit.add_argument("--json", action="store_true", help="Emit JSON report.")

    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPINFRA_API_KEY. Set it in .env or environment.")
    return api_key


def _require_telegram_credentials() -> tuple[int, str]:
    api_id_raw = os.getenv("TG_API_ID", "").strip()
    api_hash = os.getenv("TG_API_HASH", "").strip()
    if not api_id_raw:
        raise SystemExit("Missing TG_API_ID. Set it in .env or environment.")
    if not api_hash:
        raise SystemExit("Missing TG_API_HASH. Set it in .env or environment.")
    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise SystemExit("Invalid TG_API_ID. Must be an integer.") from exc
    return api_id, api_hash


def _page_out_path(pages_dir: Path, page_index: int) -> Path:
    return pages_dir / f"{page_index+1:04d}.md"


def _debug_path(debug_dir: Path, page_index: int, suffix: str) -> Path:
    return debug_dir / f"page_{page_index+1:04d}.{suffix}"


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


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
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


async def _ensure_header_ocr_artifacts(
    *,
    doc_dir: Path,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    marker_tables_path = doc_dir / "metadata" / "assets" / "structured" / "marker" / "tables_raw.jsonl"
    qa_root = doc_dir / "metadata" / "assets" / "structured" / "qa"
    ocr_out = qa_root / "bbox_ocr_outputs"
    ocr_crops = qa_root / "bbox_ocr_crops"
    ocr_out.mkdir(parents=True, exist_ok=True)
    ocr_crops.mkdir(parents=True, exist_ok=True)

    if not marker_tables_path.exists():
        return {"expected": 0, "generated": 0, "skipped_existing": 0, "failed": 0, "status": "missing_tables_raw"}

    manifest_path = doc_dir / "metadata" / "manifest.json"
    if not manifest_path.exists():
        return {"expected": 0, "generated": 0, "skipped_existing": 0, "failed": 0, "status": "missing_manifest"}
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        manifest = {}
    source_path_raw = str(manifest.get("source_path", "")).strip()
    if not source_path_raw:
        return {"expected": 0, "generated": 0, "skipped_existing": 0, "failed": 0, "status": "missing_source_path"}
    source_path = Path(source_path_raw)
    if not source_path.exists():
        return {
            "expected": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
            "status": f"missing_source_pdf:{source_path}",
        }

    table_rows = _load_jsonl_dicts(marker_tables_path)
    if not table_rows:
        return {"expected": 0, "generated": 0, "skipped_existing": 0, "failed": 0, "status": "empty_tables_raw"}

    expected = 0
    generated = 0
    skipped_existing = 0
    failed = 0
    with fitz.open(source_path) as pdf:
        by_page_counter: dict[int, int] = {}
        for row in table_rows:
            page = int(row.get("page") or 0)
            if page < 1 or page > int(pdf.page_count):
                continue
            polygons = row.get("polygons")
            bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else []
            if (not isinstance(bbox, list) or len(bbox) < 4) and isinstance(polygons, list):
                bbox = _bbox_from_polygons(polygons)
            if not bbox or len(bbox) < 4:
                continue

            by_page_counter[page] = by_page_counter.get(page, 0) + 1
            ordinal = by_page_counter[page]
            expected += 1
            out_md = ocr_out / f"table_{ordinal:02d}_page_{page:04d}.md"
            if out_md.exists():
                skipped_existing += 1
                continue

            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if x1 <= x0 or y1 <= y0:
                failed += 1
                continue
            header_h = max(72.0, min(180.0, (y1 - y0) * 0.33))
            clip = fitz.Rect(x0, y0, x1, min(y1, y0 + header_h))
            page_obj = pdf.load_page(page - 1)
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=clip, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = BytesIO()
            image.save(buf, format="PNG", optimize=True)
            image_bytes = buf.getvalue()
            (ocr_crops / f"table_{ordinal:02d}_page_{page:04d}.png").write_bytes(image_bytes)

            prompt = (
                "Extract only table header rows from this image crop as HTML table markup. "
                "Preserve Greek letters, superscripts/subscripts, symbols, and column grouping. "
                "Return only <table>...</table> with <th> cells."
            )
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
                out_md.write_text(f"<!-- header_ocr_error: {exc} -->")
                failed += 1
                continue
            out_md.write_text(str(response.content or "").strip())
            generated += 1

    status = "ok" if failed == 0 else "partial"
    return {
        "expected": expected,
        "generated": generated,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "status": status,
    }


async def _extract_bibliography(
    client: AsyncOpenAI,
    metadata_model: str,
    first_page_markdown: str,
) -> dict[str, Any]:
    if not first_page_markdown.strip():
        return {
            "title": "",
            "authors": [],
            "year": "",
            "journal_ref": "",
            "doi": "",
            "citation": "",
        }

    try:
        prompt = bibliography_prompt(first_page_markdown)
        response = await call_text_model(
            client=client,
            model=metadata_model,
            prompt=prompt,
            max_tokens=800,
        )
        raw = extract_json_object(response.content)
        parsed = normalize_bibliography(raw)
    except Exception:
        parsed = normalize_bibliography({})

    return {
        "title": parsed.title,
        "authors": parsed.authors,
        "year": parsed.year,
        "journal_ref": parsed.journal_ref,
        "doi": parsed.doi,
        "citation": citation_from_bibliography(parsed),
    }


def _final_doc_dir(args: argparse.Namespace, pdf_path: Path, bibliography: dict[str, Any]) -> Path:
    group_dir = args.out_dir / output_group_name(pdf_path)
    doc_id = doc_id_from_sha(file_sha256(pdf_path))
    return group_dir / f"doc_{doc_id}"


async def _extract_discovery(
    client: AsyncOpenAI,
    metadata_model: str,
    bibliography: dict[str, Any],
    page_count: int,
    consolidated_markdown: str,
) -> dict[str, Any]:
    if not consolidated_markdown.strip():
        return {"paper_summary": "", "key_topics": [], "sections": []}

    async def _json_call(
        prompt: str,
        max_tokens: int,
        schema: str,
        attempts: int = 4,
    ) -> dict[str, Any]:
        for _ in range(attempts):
            try:
                response = await call_text_model(
                    client=client,
                    model=metadata_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                candidates = [response.content or "", response.reasoning_content or ""]
                for c in candidates:
                    raw = extract_json_object(c)
                    if isinstance(raw, dict) and raw:
                        return raw
                reasoning = (response.reasoning_content or "").strip()
                if reasoning:
                    repair_prompt = (
                        "Convert these draft notes into one strict JSON object.\n"
                        "Output JSON only, no prose.\n"
                        f"Schema:\n{schema}\n\n"
                        f"Notes:\n{reasoning[:7000]}"
                    )
                    repaired = await call_text_model(
                        client=client,
                        model=metadata_model,
                        prompt=repair_prompt,
                        max_tokens=max_tokens,
                    )
                    for c in [repaired.content or "", repaired.reasoning_content or ""]:
                        raw = extract_json_object(c)
                        if isinstance(raw, dict) and raw:
                            return raw
            except Exception:
                continue
        return {}

    abstract_schema = '{' '"abstract":"...",' '"key_topics":["..."],' '}'
    excerpt = first_pages_excerpt(consolidated_markdown, max_pages=5)
    abstract_prompt = abstract_extraction_prompt(
        title=str(bibliography.get("title", "")),
        citation=str(bibliography.get("citation", "")),
        page_count=page_count,
        first_pages_markdown=excerpt,
    )
    abstract_raw = await _json_call(abstract_prompt, max_tokens=700, schema=abstract_schema)
    base_discovery = {"paper_summary": "", "key_topics": [], "sections": []}
    if abstract_raw:
        base_discovery = normalize_discovery(
            {
                "paper_summary": str(abstract_raw.get("abstract", "")).strip(),
                "key_topics": abstract_raw.get("key_topics", []),
                "sections": [],
            },
            page_count=page_count,
        )
    chunks = split_markdown_for_discovery(consolidated_markdown, max_chars=32000)
    chunk_payloads: list[dict[str, Any]] = []
    if chunks:
        chunk_schema = (
            '{'
            '"chunk_summary":"...",'
            '"key_topics":["..."],'
            '"sections":[{"title":"...","start_page":1,"end_page":1,"summary":"..."}]'
            '}'
        )
        for idx, chunk in enumerate(chunks, start=1):
            prompt = discoverability_chunk_prompt(
                title=str(bibliography.get("title", "")),
                citation=str(bibliography.get("citation", "")),
                page_count=page_count,
                chunk_index=idx,
                chunk_count=len(chunks),
                markdown_chunk=chunk,
            )
            raw = await _json_call(prompt, max_tokens=1000, schema=chunk_schema)
            if isinstance(raw, dict) and raw:
                chunk_payloads.append(raw)

    if chunk_payloads:
        agg_schema = (
            '{'
            '"paper_summary":"...",'
            '"key_topics":["..."],'
            '"sections":[{"title":"...","start_page":1,"end_page":1,"summary":"..."}]'
            '}'
        )
        agg_prompt = discoverability_aggregate_prompt(
            title=str(bibliography.get("title", "")),
            citation=str(bibliography.get("citation", "")),
            page_count=page_count,
            chunk_outputs=chunk_payloads,
        )
        agg_raw = await _json_call(agg_prompt, max_tokens=1200, schema=agg_schema)
        agg_discovery = normalize_discovery(agg_raw, page_count=page_count) if agg_raw else {"paper_summary": "", "key_topics": [], "sections": []}
        if agg_discovery.get("sections"):
            if not agg_discovery.get("paper_summary"):
                agg_discovery["paper_summary"] = str(base_discovery.get("paper_summary", ""))
            if not agg_discovery.get("key_topics"):
                agg_discovery["key_topics"] = list(base_discovery.get("key_topics", []))
            return agg_discovery

    if is_useful_discovery(base_discovery):
        return base_discovery
    return {"paper_summary": "", "key_topics": [], "sections": []}


def _move_first_page_artifacts(
    first_page: dict[str, Any],
    from_dirs: dict[str, Path],
    to_dirs: dict[str, Path],
    debug: bool,
) -> dict[str, Any]:
    output_files = dict(first_page.get("output_files", {}))
    page_index = int(first_page.get("page_index", 0))

    src_md = Path(output_files.get("markdown", ""))
    if src_md.exists():
        dst_md = _page_out_path(to_dirs["pages"], page_index)
        dst_md.parent.mkdir(parents=True, exist_ok=True)
        src_md.replace(dst_md)
        output_files["markdown"] = str(dst_md)

    src_meta = Path(output_files.get("metadata", ""))
    if src_meta.exists():
        dst_meta = _debug_path(to_dirs["debug"], page_index, "metadata.json")
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        src_meta.replace(dst_meta)
        output_files["metadata"] = str(dst_meta)

    if debug:
        for suffix in ("request.json", "response.json"):
            src = _debug_path(from_dirs["debug"], page_index, suffix)
            if src.exists():
                dst = _debug_path(to_dirs["debug"], page_index, suffix)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)

    # Move structured page assets that may have been written to staging assets.
    page_tag = f"page_{page_index + 1:04d}"
    for root in ("marker", "grobid"):
        src_root = from_dirs["assets"] / "structured" / root
        if not src_root.exists():
            continue
        dst_root = to_dirs["assets"] / "structured" / root
        for candidate in src_root.glob(f"{page_tag}*"):
            dst = dst_root / candidate.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if candidate.is_dir():
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(candidate, dst)
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.replace(dst)

    first_page["output_files"] = output_files
    return first_page


def _collect_page_signals(doc: fitz.Document, mode: str) -> list[tuple[Any, str]]:
    signals: list[tuple[Any, str]] = []
    for i in range(doc.page_count):
        page_dict = doc.load_page(i).get_text("dict")
        heuristics = compute_text_heuristics(page_dict)
        route = decide_route(heuristics, mode=mode)
        signals.append((heuristics, route))
    return signals


def _render_dims_for_route(page_width_pt: float, page_height_pt: float, route: str, max_dim: int = 1288) -> tuple[int, int]:
    dpi = 200 if route == "anchored" else 300
    # Keep manifest render dims aligned with render._resize_longest truncation semantics.
    w = max(int(page_width_pt * dpi / 72.0), 1)
    h = max(int(page_height_pt * dpi / 72.0), 1)
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        w = max(int(w * scale), 1)
        h = max(int(h * scale), 1)
    return w, h


def _job_slug_from_csv_stem(stem: str) -> str:
    slug = CSV_JOB_SAFE_RE.sub("_", stem).strip("._-").lower().replace(".", "_")
    slug = UNDERSCORE_RUN_RE.sub("_", slug).strip("_")
    return slug or "job"


def _is_same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:  # noqa: BLE001
        return str(a) == str(b)


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:  # noqa: BLE001
        return False


def _resolve_fetch_job_dir(output_root: Path, csv_slug: str) -> Path:
    job_dir = output_root / csv_slug
    if _is_same_path(output_root, DEFAULT_FETCH_OUTPUT_ROOT):
        legacy_job_dir = LEGACY_FETCH_OUTPUT_ROOT / csv_slug
        if legacy_job_dir.exists() and not job_dir.exists():
            job_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_job_dir), str(job_dir))
            print(
                "[fetch-telegram] migrated legacy job folder "
                f"{legacy_job_dir} -> {job_dir}"
            )
    return job_dir


async def _process_page_structured(
    *,
    page_index: int,
    pdf_path: Path,
    marker_command: str,
    marker_url: str,
    marker_timeout: int,
    structured_asset_level: str,
    structured_backend: str,
    structured_semaphore: asyncio.Semaphore,
    route: str,
    heuristics: Any,
    fallback_kwargs: dict[str, Any],
) -> dict[str, Any]:
    dirs = fallback_kwargs["dirs"]
    page_path = _page_out_path(dirs["pages"], page_index)
    if page_path.exists() and not fallback_kwargs.get("force", False):
        return {
            "page_index": page_index,
            "route": route,
            "heuristics": asdict(heuristics),
            "status": "skipped",
            "output_files": {"markdown": str(page_path)},
        }

    async with structured_semaphore:
        marker_result = await asyncio.to_thread(
            run_marker_page,
            pdf_path,
            page_index,
            marker_command,
            marker_timeout,
            dirs["assets"],
            structured_asset_level,
            marker_url,
        )

    if marker_result.success:
        md = normalize_markdown_for_llm(marker_result.markdown)
        write_text(page_path, md)
        metadata_path = _debug_path(dirs["debug"], page_index, "metadata.json")
        write_json(
            metadata_path,
            {
                "mode": "structured",
                "backend": structured_backend,
                "artifacts": marker_result.artifacts,
            },
        )
        return {
            "page_index": page_index,
            "route": route,
            "heuristics": asdict(heuristics),
            "attempts": 0,
            "token_usage": None,
            "status": "structured_ok",
            "output_files": {
                "markdown": str(page_path),
                "metadata": str(metadata_path),
            },
            "structured": {
                "backend": structured_backend,
                "artifacts": marker_result.artifacts,
                "fallback_reason": "",
            },
        }

    fallback = await _process_page(**fallback_kwargs, route_override=route, heuristics_override=heuristics)
    fallback["status"] = "structured_fallback"
    fallback["structured"] = {
        "backend": structured_backend,
        "artifacts": {},
        "fallback_reason": marker_result.error or "structured extraction failed",
    }
    return fallback


async def _process_page(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    doc: fitz.Document,
    page_index: int,
    mode: str,
    max_tokens: int,
    model: str,
    scan_preprocess: bool,
    debug: bool,
    dirs: dict[str, Path],
    force: bool,
    text_only_enabled: bool,
    route_override: str | None = None,
    heuristics_override: Any | None = None,
) -> dict[str, Any]:
    page = doc.load_page(page_index)
    page_dict = page.get_text("dict")
    heuristics = heuristics_override or compute_text_heuristics(page_dict)
    route = route_override or decide_route(heuristics, mode=mode)

    page_path = _page_out_path(dirs["pages"], page_index)
    if page_path.exists() and not force:
        return {
            "page_index": page_index,
            "route": route,
            "heuristics": asdict(heuristics),
            "status": "skipped",
            "output_files": {"markdown": str(page_path)},
        }

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    text_only = text_only_enabled and route == "anchored" and is_text_only_candidate(heuristics)

    if text_only:
        text = page.get_text("text")
        write_text(page_path, normalize_markdown_for_llm(text))
        metadata_path = _debug_path(dirs["debug"], page_index, "metadata.json")
        write_json(metadata_path, {"mode": "text_only"})
        return {
            "page_index": page_index,
            "route": route,
            "heuristics": asdict(heuristics),
            "attempts": 0,
            "token_usage": None,
            "status": "text_only",
            "output_files": {
                "markdown": str(page_path),
                "metadata": str(metadata_path),
            },
        }

    if route == "anchored":
        anchor_payload = extract_anchors(page_dict, page_width, page_height)
        prompt, prompt_builder = build_anchored_prompt(anchor_payload)
    else:
        prompt, prompt_builder = build_unanchored_prompt(page_width, page_height)

    render = render_page(page, route=route, scan_preprocess=scan_preprocess)

    request_payload = {
        "model": model,
        "max_tokens": max_tokens,
        "prompt_builder": prompt_builder,
        "prompt": prompt,
        "image_format": render.format,
    }

    async with semaphore:
        response = await call_olmocr(
            client=client,
            model=model,
            prompt=prompt,
            image_bytes=render.image_bytes,
            mime_type=render.mime_type,
            max_tokens=max_tokens,
        )

    parsed = parse_yaml_front_matter(response.content)

    # Rotation retry logic
    metadata = parsed.metadata or {}
    is_rotation_valid = metadata.get("is_rotation_valid", True)
    rotation_correction = metadata.get("rotation_correction", 0)
    attempts = 1
    if not is_rotation_valid and rotation_correction:
        attempts += 1
        render = render_page(
            page,
            route=route,
            scan_preprocess=scan_preprocess,
            rotation=int(rotation_correction),
        )
        async with semaphore:
            response = await call_olmocr(
                client=client,
                model=model,
                prompt=prompt,
                image_bytes=render.image_bytes,
                mime_type=render.mime_type,
                max_tokens=max_tokens,
            )
        parsed = parse_yaml_front_matter(response.content)
        metadata = parsed.metadata or {}
        is_rotation_valid = metadata.get("is_rotation_valid", True)

    if not is_rotation_valid:
        for rot in (90, 180, 270):
            attempts += 1
            render = render_page(
                page,
                route=route,
                scan_preprocess=scan_preprocess,
                rotation=rot,
            )
            async with semaphore:
                response = await call_olmocr(
                    client=client,
                    model=model,
                    prompt=prompt,
                    image_bytes=render.image_bytes,
                    mime_type=render.mime_type,
                    max_tokens=max_tokens,
                )
            parsed = parse_yaml_front_matter(response.content)
            metadata = parsed.metadata or {}
            is_rotation_valid = metadata.get("is_rotation_valid", True)
            if is_rotation_valid:
                break

    write_text(page_path, normalize_markdown_for_llm(parsed.markdown))
    metadata_path = _debug_path(dirs["debug"], page_index, "metadata.json")
    write_json(metadata_path, metadata)

    if debug:
        write_json(_debug_path(dirs["debug"], page_index, "request.json"), request_payload)
        write_json(_debug_path(dirs["debug"], page_index, "response.json"), response.raw)

    return {
        "page_index": page_index,
        "route": route,
        "heuristics": asdict(heuristics),
        "attempts": attempts,
        "token_usage": response.usage,
        "status": "ok",
        "output_files": {
            "markdown": str(page_path),
            "metadata": str(metadata_path),
        },
    }


def _merge_bibliography_with_patch(
    bibliography: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(bibliography)
    for key in ("title", "year", "doi", "journal_ref"):
        if not str(merged.get(key, "")).strip() and str(patch.get(key, "")).strip():
            merged[key] = str(patch.get(key, "")).strip()
    if not merged.get("authors") and isinstance(patch.get("authors"), list):
        merged["authors"] = [str(a).strip() for a in patch.get("authors", []) if str(a).strip()]
    merged["citation"] = citation_from_bibliography(normalize_bibliography(merged))
    return merged


async def _process_pdf(args: argparse.Namespace, pdf_path: Path) -> dict[str, Any]:
    sha = file_sha256(pdf_path)
    doc_id = doc_id_from_sha(sha)
    group_dir = args.out_dir / output_group_name(pdf_path)
    doc_dir = group_dir / f"doc_{doc_id}"
    existing_manifest_path = doc_dir / "metadata" / "manifest.json"
    if existing_manifest_path.exists() and not bool(getattr(args, "force", False)):
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text())
        except Exception:
            existing_manifest = {}
        existing_sha = str(existing_manifest.get("sha256", "")).strip()
        if existing_sha and existing_sha == sha:
            bib_path = doc_dir / "metadata" / "bibliography.json"
            disc_path = doc_dir / "metadata" / "discovery.json"
            try:
                bibliography = json.loads(bib_path.read_text()) if bib_path.exists() else existing_manifest.get("bibliography", {})
            except Exception:
                bibliography = existing_manifest.get("bibliography", {})
            try:
                discovery = json.loads(disc_path.read_text()) if disc_path.exists() else existing_manifest.get("discovery", {})
            except Exception:
                discovery = existing_manifest.get("discovery", {})
            return {
                "group_dir": str(group_dir),
                "doc_dir": str(doc_dir),
                "folder_name": doc_dir.name,
                "consolidated_markdown": "document.md",
                "page_count": int(existing_manifest.get("page_count", 0) or 0),
                "bibliography": bibliography if isinstance(bibliography, dict) else {},
                "discovery": discovery if isinstance(discovery, dict) else {"paper_summary": "", "key_topics": [], "sections": []},
                "status": "skipped",
                "strict_failure": False,
                "errors": [],
            }
    staging_dir = group_dir / f".staging_{doc_id}"
    dirs = ensure_dirs(staging_dir)

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        manifest = new_manifest(
            doc_id=doc_id,
            source_path=str(pdf_path),
            sha256=sha,
            page_count=page_count,
            model=args.model,
            base_url=args.base_url,
            prompt_version="v1",
        )

        api_key = _require_api_key()
        client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
        semaphore = asyncio.Semaphore(args.workers)
        structured_max_workers = max(1, int(getattr(args, "structured_max_workers", 4) or 1))
        digital_structured = str(getattr(args, "digital_structured", "off"))
        structured_backend = str(getattr(args, "structured_backend", "hybrid"))
        marker_command = str(getattr(args, "marker_command", MARKER_COMMAND_DEFAULT))
        marker_url = str(getattr(args, "marker_url", MARKER_URL_DEFAULT) or "")
        marker_timeout = int(getattr(args, "marker_timeout", int(MARKER_TIMEOUT_DEFAULT)))
        structured_asset_level = str(getattr(args, "structured_asset_level", "standard"))
        grobid_url = str(getattr(args, "grobid_url", "") or "")
        grobid_timeout = int(getattr(args, "grobid_timeout", int(GROBID_TIMEOUT_DEFAULT)))

        structured_semaphore = asyncio.Semaphore(structured_max_workers)
        page_signals = _collect_page_signals(doc, args.mode) if page_count > 0 else []
        routes = [route for _, route in page_signals]
        heuristics_by_page = [heuristics for heuristics, _ in page_signals]
        page_contracts: dict[int, dict[str, Any]] = {}
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            route = routes[page_index] if routes else "unanchored"
            render_w, render_h = _render_dims_for_route(float(page.rect.width), float(page.rect.height), route)
            page_contracts[page_index + 1] = build_render_contract(
                pdf_page_w_pt=float(page.rect.width),
                pdf_page_h_pt=float(page.rect.height),
                render_page_w_px=render_w,
                render_page_h_px=render_h,
                rotation_degrees=0,
            )
        structured_enabled = is_structured_candidate_doc(
            digital_structured,
            routes,
            heuristics_by_page,
        )
        grobid_used = False
        grobid_error = ""
        grobid_sections: list[dict[str, Any]] = []
        grobid_figures_tables_count = 0

        async def _process_one_page(page_index: int, page_dirs: dict[str, Path]) -> dict[str, Any]:
            heuristics = heuristics_by_page[page_index] if page_signals else compute_text_heuristics(doc.load_page(page_index).get_text("dict"))
            route = routes[page_index] if page_signals else decide_route(heuristics, mode=args.mode)
            fallback_kwargs = {
                "semaphore": semaphore,
                "client": client,
                "doc": doc,
                "page_index": page_index,
                "mode": args.mode,
                "max_tokens": args.max_tokens,
                "model": args.model,
                "scan_preprocess": args.scan_preprocess,
                "debug": args.debug,
                "dirs": page_dirs,
                "force": args.force,
                "text_only_enabled": args.text_only,
            }
            if structured_enabled:
                return await _process_page_structured(
                    page_index=page_index,
                    pdf_path=pdf_path,
                    marker_command=marker_command,
                    marker_url=marker_url,
                    marker_timeout=marker_timeout,
                    structured_asset_level=structured_asset_level,
                    structured_backend=structured_backend,
                    structured_semaphore=structured_semaphore,
                    route=route,
                    heuristics=heuristics,
                    fallback_kwargs=fallback_kwargs,
                )
            return await _process_page(
                **fallback_kwargs,
                route_override=route,
                heuristics_override=heuristics,
            )

        pages: list[dict[str, Any]] = []
        if page_count > 0:
            first_page = await _process_one_page(0, dirs)
            first_page_md = Path(first_page["output_files"].get("markdown", ""))
            first_page_text = first_page_md.read_text() if first_page_md.exists() else ""
            bibliography = await _extract_bibliography(client, args.metadata_model, first_page_text)
            if structured_enabled and structured_backend == "hybrid" and grobid_url.strip():
                grobid_result = await asyncio.to_thread(
                    run_grobid_doc,
                    pdf_path,
                    grobid_url,
                    grobid_timeout,
                    dirs["assets"] / "structured" / "grobid" / "fulltext.tei.xml",
                    doc_id,
                )
                if grobid_result.success:
                    grobid_used = True
                    grobid_sections = grobid_result.sections
                    grobid_figures_tables_count = len(grobid_result.figures_tables)
                    for row in grobid_result.figures_tables:
                        coords = row.get("coords")
                        if isinstance(coords, list):
                            row["coords_px"] = grobid_coords_to_px(coords, page_contracts=page_contracts)
                    bibliography = _merge_bibliography_with_patch(bibliography, grobid_result.bibliography_patch)
                else:
                    grobid_error = grobid_result.error
            final_dirs = ensure_dirs(doc_dir)
            marker_localization = {"enabled": bool(getattr(args, "marker_localize", True))}
            if bool(getattr(args, "marker_localize", True)) and page_count > 0:
                marker_doc = await asyncio.to_thread(
                    run_marker_doc,
                    pdf_path,
                    marker_command,
                    marker_timeout,
                    final_dirs["assets"],
                    str(getattr(args, "marker_localize_profile", "full_json")),
                    marker_url,
                )
                marker_localization.update(
                    {
                        "success": marker_doc.success,
                        "error": marker_doc.error,
                        "artifacts": marker_doc.artifacts,
                        "localization_page_status": marker_doc.localization_page_status,
                    }
                )
            elif bool(getattr(args, "marker_localize", True)):
                marker_localization.update(
                    {
                        "success": False,
                        "error": "skipped_empty_document",
                        "artifacts": {},
                        "localization_page_status": {},
                    }
                )
            first_page = _move_first_page_artifacts(first_page, dirs, final_dirs, args.debug)
            src_tei = dirs["assets"] / "structured" / "grobid" / "fulltext.tei.xml"
            if src_tei.exists():
                dst_tei = final_dirs["assets"] / "structured" / "grobid" / "fulltext.tei.xml"
                dst_tei.parent.mkdir(parents=True, exist_ok=True)
                src_tei.replace(dst_tei)
            pages.append(first_page)
        else:
            bibliography = {
                "title": "",
                "authors": [],
                "year": "",
                "journal_ref": "",
                "doi": "",
                "citation": "",
            }
            final_dirs = ensure_dirs(doc_dir)
            marker_localization = {"enabled": bool(getattr(args, "marker_localize", True))}
            if bool(getattr(args, "marker_localize", True)):
                marker_doc = await asyncio.to_thread(
                    run_marker_doc,
                    pdf_path,
                    marker_command,
                    marker_timeout,
                    final_dirs["assets"],
                    str(getattr(args, "marker_localize_profile", "full_json")),
                    marker_url,
                )
                marker_localization.update(
                    {
                        "success": marker_doc.success,
                        "error": marker_doc.error,
                        "artifacts": marker_doc.artifacts,
                        "localization_page_status": marker_doc.localization_page_status,
                    }
                )

        tasks = [_process_one_page(i, final_dirs) for i in range(1, page_count)]

        if tasks:
            pages.extend(await asyncio.gather(*tasks))
        pages_sorted = sorted(pages, key=lambda p: p["page_index"])
        for p in pages_sorted:
            manifest["pages"].append(p)
        manifest["bibliography"] = bibliography
        structured_page_count = sum(1 for p in pages_sorted if p.get("status") == "structured_ok")
        fallback_count = sum(1 for p in pages_sorted if p.get("status") == "structured_fallback")
        structured_manifest: dict[str, Any] = {
            "enabled": structured_enabled,
            "backend": structured_backend if structured_enabled else "none",
            "grobid_used": grobid_used,
            "grobid_figures_tables_count": grobid_figures_tables_count,
            "fallback_count": fallback_count,
            "structured_page_count": structured_page_count,
            "marker_localization": marker_localization,
        }
        if grobid_error:
            structured_manifest["grobid_error"] = grobid_error
        manifest["structured_extraction"] = structured_manifest
        manifest["render_contract"] = {str(k): v for k, v in page_contracts.items()}
        marker_cfg_hash = hashlib.sha1(
            f"{marker_command}|{marker_url}|{getattr(args, 'marker_localize_profile', 'full_json')}".encode("utf-8")
        ).hexdigest()
        runtime = manifest.get("runtime", {}) if isinstance(manifest.get("runtime", {}), dict) else {}
        runtime.update(
            {
                "pipeline_version": "v2-table-pipeline",
                "marker_version": str(marker_localization.get("artifacts", {}).get("raw_json", "")),
                "marker_config_hash": marker_cfg_hash,
                "grobid_version": "unknown",
                "deterministic_params": {"temperature": 0},
                "source_sha_match": True,
                "rerun_policy": "skip_if_manifest_sha_match_unless_force",
            }
        )
        manifest["runtime"] = runtime

        write_json(final_dirs["metadata"] / "manifest.json", manifest)
        write_json(final_dirs["metadata"] / "bibliography.json", bibliography)

        # Assemble document outputs
        md_out = []
        jsonl_out = []
        for p in pages_sorted:
            md_path = Path(p["output_files"].get("markdown", ""))
            if not md_path.exists():
                continue
            md = md_path.read_text()
            md_out.append(f"\n\n# Page {p['page_index'] + 1}\n\n{md}\n")
            jsonl_out.append(
                json.dumps(
                    {
                        "page_index": p["page_index"],
                        "route": p.get("route"),
                        "markdown": md,
                        "metadata_path": p["output_files"].get("metadata"),
                    },
                    ensure_ascii=True,
                )
            )

        consolidated_name = "document.md"
        consolidated_text = "".join(md_out).strip() + "\n"
        write_text(doc_dir / consolidated_name, consolidated_text)
        write_text(final_dirs["metadata"] / "document.jsonl", "\n".join(jsonl_out) + "\n")
        discovery = await _extract_discovery(
            client=client,
            metadata_model=args.metadata_model,
            bibliography=bibliography,
            page_count=page_count,
            consolidated_markdown=consolidated_text,
        )
        if grobid_sections:
            discovered_sections = discovery.get("sections", []) or []
            grobid_high_conf = (
                len(grobid_sections) >= 3
                or any(
                    int(sec.get("start_page", 1) or 1) > 1 or int(sec.get("end_page", 1) or 1) > 1
                    for sec in grobid_sections
                    if isinstance(sec, dict)
                )
            )
            if grobid_high_conf or not discovered_sections:
                discovery["sections"] = grobid_sections
        write_json(final_dirs["metadata"] / "discovery.json", discovery)
        write_json(final_dirs["metadata"] / "sections.json", discovery.get("sections", []))
        structured_data_enabled = bool(getattr(args, "extract_structured_data", True))
        deplot_command = str(getattr(args, "deplot_command", "") or "")
        deplot_timeout = int(getattr(args, "deplot_timeout", int(DEPLOT_TIMEOUT_DEFAULT)))
        structured_data_manifest: dict[str, Any] = {
            "enabled": structured_data_enabled,
            "table_count": 0,
            "figure_count": 0,
            "deplot_count": 0,
            "unresolved_figure_count": 0,
            "errors": [],
            "header_ocr": {},
            "ocr_merge": {},
            "ocr_html_comparison": {},
        }
        strict_failure = False
        doc_errors: list[str] = []
        if structured_data_enabled:
            try:
                if bool(getattr(args, "table_ocr_merge", True)) and bool(getattr(args, "table_header_ocr_auto", True)):
                    header_ocr = await _ensure_header_ocr_artifacts(
                        doc_dir=doc_dir,
                        client=client,
                        model=str(getattr(args, "table_header_ocr_model", TABLE_HEADER_OCR_MODEL_DEFAULT)),
                        max_tokens=int(getattr(args, "table_header_ocr_max_tokens", TABLE_HEADER_OCR_MAX_TOKENS_DEFAULT)),
                    )
                    structured_data_manifest["header_ocr"] = header_ocr
                summary = build_structured_exports(
                    doc_dir=doc_dir,
                    deplot_command=deplot_command,
                    deplot_timeout=deplot_timeout,
                    table_source=str(getattr(args, "table_source", "marker-first")),
                    table_ocr_merge=bool(getattr(args, "table_ocr_merge", True)),
                    table_ocr_merge_scope=str(getattr(args, "table_ocr_merge_scope", "header")),
                    table_artifact_mode=str(getattr(args, "table_artifact_mode", "permissive")),
                    ocr_html_dir=getattr(args, "ocr_html_dir", None),
                    table_quality_gate=bool(getattr(args, "table_quality_gate", True)),
                    table_escalation=str(getattr(args, "table_escalation", "auto")),
                    table_escalation_max=int(getattr(args, "table_escalation_max", 20)),
                    table_qa_mode=str(getattr(args, "table_qa_mode", "warn")),
                    grobid_status="ok" if grobid_used else ("error" if grobid_error else "missing"),
                )
                structured_data_manifest.update(
                    {
                        "table_count": summary.table_count,
                        "figure_count": summary.figure_count,
                        "deplot_count": summary.deplot_count,
                        "unresolved_figure_count": summary.unresolved_figure_count,
                        "errors": summary.errors,
                        "ocr_merge": summary.ocr_merge,
                    }
                )
                if bool(getattr(args, "compare_ocr_html", False)):
                    ocr_html_dir = getattr(args, "ocr_html_dir", None)
                    compare_summary = compare_marker_tables_with_ocr_html(
                        doc_dir=doc_dir,
                        ocr_html_dir=ocr_html_dir,
                    )
                    structured_data_manifest["ocr_html_comparison"] = compare_summary
            except Exception as exc:  # noqa: BLE001
                structured_data_manifest["errors"] = [str(exc)]
                structured_data_manifest["ocr_merge"] = {}
                if "Strict table artifact mode failed" in str(exc):
                    strict_failure = True
                doc_errors.append(str(exc))
        manifest["structured_data_extraction"] = structured_data_manifest
        qa_flags_path = final_dirs["assets"] / "structured" / "extracted" / "qa" / "table_flags.jsonl"
        if not qa_flags_path.exists():
            qa_flags_path = final_dirs["assets"] / "structured" / "qa" / "table_flags.jsonl"
        qa_flag_count = 0
        if qa_flags_path.exists():
            qa_flag_count = len([ln for ln in qa_flags_path.read_text().splitlines() if ln.strip()])
        manifest["table_pipeline"] = {
            "enabled": structured_data_enabled,
            "marker_localized": bool(marker_localization.get("success")),
            "qa_mode": str(getattr(args, "table_qa_mode", "warn")),
            "qa_flags": qa_flag_count,
        }
        manifest["table_qa"] = {
            "mode": str(getattr(args, "table_qa_mode", "warn")),
            "status": "ok" if qa_flag_count == 0 else "flags",
            "qa_skipped_reason": "" if grobid_used else ("grobid_error" if grobid_error else "grobid_not_run"),
        }
        manifest["processing_status"] = {
            "status": "failed" if strict_failure else "ok",
            "errors": doc_errors,
            "strict_failure": strict_failure,
        }
        manifest["discovery"] = discovery
        write_json(final_dirs["metadata"] / "manifest.json", manifest)

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    return {
        "group_dir": str(group_dir),
        "doc_dir": str(doc_dir),
        "folder_name": doc_dir.name,
        "consolidated_markdown": consolidated_name,
        "page_count": page_count,
        "bibliography": bibliography,
        "discovery": discovery,
        "status": "failed" if strict_failure else "ok",
        "strict_failure": strict_failure,
        "errors": doc_errors,
    }


def _write_group_readmes(records: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        group_key = str(rec.get("group_dir", ""))
        by_group.setdefault(group_key, []).append(rec)

    for group_str, papers in by_group.items():
        group_dir = Path(group_str)
        group_name = group_dir.name
        readme_text = render_group_readme(group_name, papers)
        write_text(group_dir / "README.md", readme_text)


async def _run(args: argparse.Namespace) -> None:
    if _is_path_within(args.out_dir, DEFAULT_FETCH_OUTPUT_ROOT):
        raise SystemExit(
            "Refusing to write final OCR outputs under data/jobs. "
            "Use a canonical output folder such as out/<job_slug>."
        )
    pdfs = discover_pdfs(args.in_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failed_docs = 0
    for pdf in pdfs:
        try:
            rec = await _process_pdf(args, pdf)
        except Exception as exc:  # noqa: BLE001
            rec = {
                "group_dir": str(args.out_dir / output_group_name(pdf)),
                "doc_dir": "",
                "folder_name": "",
                "consolidated_markdown": "",
                "page_count": 0,
                "bibliography": {},
                "discovery": {},
                "status": "failed",
                "strict_failure": True,
                "errors": [str(exc)],
            }
        records.append(rec)
        if str(rec.get("status", "ok")) == "failed":
            failed_docs += 1
    _write_group_readmes(records)
    if failed_docs > 0:
        raise SystemExit(f"Run completed with failed_docs={failed_docs}")


async def _run_fetch_telegram(args: argparse.Namespace) -> None:
    if not args.doi_csv.exists():
        raise SystemExit(f"DOI CSV does not exist: {args.doi_csv}")
    if not str(args.target_bot).strip():
        raise SystemExit("Missing target bot. Set TARGET_BOT in .env or pass --target-bot.")

    api_id, api_hash = _require_telegram_credentials()
    csv_name = _job_slug_from_csv_stem(args.doi_csv.stem)
    job_dir = _resolve_fetch_job_dir(args.output_root, csv_name)
    pdf_dir = job_dir / "pdfs"
    reports_dir = job_dir / "reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = FetchTelegramConfig(
        api_id=api_id,
        api_hash=api_hash,
        doi_csv=args.doi_csv,
        in_dir=pdf_dir,
        doi_column=args.doi_column,
        target_bot=args.target_bot,
        session_name=args.session_name,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        response_timeout=args.response_timeout,
        search_timeout=args.search_timeout,
        report_file=args.report_file or (reports_dir / "telegram_download_report.csv"),
        failed_file=args.failed_file or (reports_dir / "telegram_failed_papers.csv"),
        debug=args.debug,
    )
    await fetch_from_telegram(config)


def _discover_ocr_doc_dirs(ocr_out_dir: Path) -> list[Path]:
    if (ocr_out_dir / "pages").is_dir() and (ocr_out_dir / "metadata").is_dir():
        return [ocr_out_dir]

    out: list[Path] = []
    seen: set[str] = set()
    for manifest_path in sorted(ocr_out_dir.rglob("metadata/manifest.json")):
        doc_dir = manifest_path.parent.parent
        if not (doc_dir / "pages").is_dir():
            continue
        key = str(doc_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(doc_dir)
    return out


def _run_export_structured_data(args: argparse.Namespace) -> dict[str, int]:
    if not args.ocr_out_dir.exists():
        raise SystemExit(f"OCR output directory does not exist: {args.ocr_out_dir}")

    docs = _discover_ocr_doc_dirs(args.ocr_out_dir)
    if not docs:
        raise SystemExit(f"No OCR document folders found under: {args.ocr_out_dir}")

    totals = {
        "docs_processed": 0,
        "table_count": 0,
        "figure_count": 0,
        "deplot_count": 0,
    }
    failed_docs = 0

    for doc_dir in docs:
        metadata_dir = doc_dir / "metadata"
        manifest_path = metadata_dir / "manifest.json"
        manifest: dict[str, Any]
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text())
                manifest = loaded if isinstance(loaded, dict) else {}
            except Exception:
                manifest = {}
        else:
            manifest = {}

        structured_extraction = manifest.get("structured_extraction", {})
        if not isinstance(structured_extraction, dict):
            structured_extraction = {}
        if bool(structured_extraction.get("grobid_used")):
            grobid_status = "ok"
        elif str(structured_extraction.get("grobid_error", "")).strip():
            grobid_status = "error"
        elif structured_extraction:
            grobid_status = "missing"
        else:
            grobid_status = "unknown"

        try:
            header_ocr_summary: dict[str, Any] = {}
            if bool(getattr(args, "table_ocr_merge", True)) and bool(getattr(args, "table_header_ocr_auto", True)):
                api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
                if api_key:
                    base_url = str(manifest.get("base_url", "") or "https://api.deepinfra.com/v1/openai")
                    ocr_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                    header_ocr_summary = asyncio.run(
                        _ensure_header_ocr_artifacts(
                            doc_dir=doc_dir,
                            client=ocr_client,
                            model=str(getattr(args, "table_header_ocr_model", TABLE_HEADER_OCR_MODEL_DEFAULT)),
                            max_tokens=int(getattr(args, "table_header_ocr_max_tokens", TABLE_HEADER_OCR_MAX_TOKENS_DEFAULT)),
                        )
                    )
                else:
                    header_ocr_summary = {"generated": 0, "skipped_existing": 0, "status": "missing_api_key"}

            summary = build_structured_exports(
                doc_dir=doc_dir,
                deplot_command=str(getattr(args, "deplot_command", "") or ""),
                deplot_timeout=int(getattr(args, "deplot_timeout", int(DEPLOT_TIMEOUT_DEFAULT))),
                table_source=str(getattr(args, "table_source", "marker-first")),
                table_ocr_merge=bool(getattr(args, "table_ocr_merge", True)),
                table_ocr_merge_scope=str(getattr(args, "table_ocr_merge_scope", "header")),
                table_artifact_mode=str(getattr(args, "table_artifact_mode", "permissive")),
                ocr_html_dir=getattr(args, "ocr_html_dir", None),
                table_quality_gate=bool(getattr(args, "table_quality_gate", True)),
                table_escalation=str(getattr(args, "table_escalation", "auto")),
                table_escalation_max=int(getattr(args, "table_escalation_max", 20)),
                table_qa_mode=str(getattr(args, "table_qa_mode", "warn")),
                grobid_status=grobid_status,
            )
            structured_data_manifest: dict[str, Any] = {
                "enabled": True,
                "table_count": summary.table_count,
                "figure_count": summary.figure_count,
                "deplot_count": summary.deplot_count,
                "unresolved_figure_count": summary.unresolved_figure_count,
                "errors": summary.errors,
                "header_ocr": header_ocr_summary,
                "ocr_merge": summary.ocr_merge,
                "ocr_html_comparison": {},
            }
            if bool(getattr(args, "compare_ocr_html", False)):
                compare_summary = compare_marker_tables_with_ocr_html(
                    doc_dir=doc_dir,
                    ocr_html_dir=getattr(args, "ocr_html_dir", None),
                )
                structured_data_manifest["ocr_html_comparison"] = compare_summary
        except Exception as exc:  # noqa: BLE001
            structured_data_manifest = {
                "enabled": True,
                "table_count": 0,
                "figure_count": 0,
                "deplot_count": 0,
                "unresolved_figure_count": 0,
                "errors": [str(exc)],
                "header_ocr": {},
                "ocr_merge": {},
                "ocr_html_comparison": {},
            }
            if "Strict table artifact mode failed" in str(exc):
                failed_docs += 1
        manifest["structured_data_extraction"] = structured_data_manifest
        manifest["processing_status"] = {
            "status": "failed" if structured_data_manifest.get("errors") else "ok",
            "errors": list(structured_data_manifest.get("errors", [])),
            "strict_failure": any("Strict table artifact mode failed" in str(e) for e in structured_data_manifest.get("errors", [])),
        }
        write_json(manifest_path, manifest)

        totals["docs_processed"] += 1
        totals["table_count"] += int(structured_data_manifest.get("table_count", 0))
        totals["figure_count"] += int(structured_data_manifest.get("figure_count", 0))
        totals["deplot_count"] += int(structured_data_manifest.get("deplot_count", 0))

        print(
            "[structured-export] "
            f"doc={doc_dir} "
            f"tables={structured_data_manifest['table_count']} "
            f"figures={structured_data_manifest['figure_count']} "
            f"deplot={structured_data_manifest['deplot_count']}"
        )

    print(
        "[structured-export] done "
        f"docs={totals['docs_processed']} "
        f"tables={totals['table_count']} "
        f"figures={totals['figure_count']} "
        f"deplot={totals['deplot_count']}"
    )
    if failed_docs > 0:
        raise SystemExit(f"Structured export completed with failed_docs={failed_docs}")
    return totals


def _run_export_facts(args: argparse.Namespace) -> dict[str, int]:
    if not args.ocr_out_dir.exists():
        raise SystemExit(f"OCR output directory does not exist: {args.ocr_out_dir}")
    docs = _discover_ocr_doc_dirs(args.ocr_out_dir)
    if not docs:
        raise SystemExit(f"No OCR document folders found under: {args.ocr_out_dir}")

    totals = {
        "docs_processed": 0,
        "record_count": 0,
    }
    for doc_dir in docs:
        summary = export_facts_for_doc(doc_dir)
        totals["docs_processed"] += 1
        totals["record_count"] += int(summary.get("record_count", 0) or 0)
        print(f"[facts-export] doc={doc_dir} records={summary.get('record_count', 0)}")
    print(f"[facts-export] done docs={totals['docs_processed']} records={totals['record_count']}")
    return totals


def _run_eval_table_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    metrics = evaluate_table_pipeline(args.gold_dir, args.pred_dir)
    baseline_path = getattr(args, "baseline", None)
    strict_regression = bool(getattr(args, "strict_regression", False))
    if baseline_path:
        try:
            baseline = json.loads(Path(baseline_path).read_text())
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid baseline metrics file: {baseline_path} ({exc})") from exc
        if not isinstance(baseline, dict):
            raise SystemExit(f"Invalid baseline metrics file: {baseline_path}")
        precision_drop = float(baseline.get("table_detection_precision", 0.0)) - float(metrics.get("table_detection_precision", 0.0))
        recall_drop = float(baseline.get("table_detection_recall", 0.0)) - float(metrics.get("table_detection_recall", 0.0))
        numeric_parse_success = float(metrics.get("numeric_parse_success", 0.0))
        max_precision_drop = float(getattr(args, "max_precision_drop", 0.03))
        max_recall_drop = float(getattr(args, "max_recall_drop", 0.03))
        min_numeric_parse = float(getattr(args, "min_numeric_parse", 0.8))
        regression = {
            "baseline_path": str(Path(baseline_path)),
            "precision_drop": precision_drop,
            "recall_drop": recall_drop,
            "numeric_parse_success": numeric_parse_success,
            "max_precision_drop": max_precision_drop,
            "max_recall_drop": max_recall_drop,
            "min_numeric_parse": min_numeric_parse,
            "violations": [],
        }
        if precision_drop > max_precision_drop:
            regression["violations"].append(f"precision_drop={precision_drop:.4f} > {max_precision_drop:.4f}")
        if recall_drop > max_recall_drop:
            regression["violations"].append(f"recall_drop={recall_drop:.4f} > {max_recall_drop:.4f}")
        if numeric_parse_success < min_numeric_parse:
            regression["violations"].append(f"numeric_parse_success={numeric_parse_success:.4f} < {min_numeric_parse:.4f}")
        metrics["regression"] = regression
        if strict_regression and regression["violations"]:
            print(json.dumps(metrics, indent=2))
            raise SystemExit("Regression checks failed")
    print(json.dumps(metrics, indent=2))
    return metrics


def _run_data_audit(args: argparse.Namespace) -> dict[str, Any]:
    report = run_data_audit(args.data_dir)
    payload = report.to_dict()
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2))
    else:
        print(format_audit_report(report))
    if report.issue_count > 0 and bool(getattr(args, "strict", False)):
        raise SystemExit(f"Data layout contract violations: {report.issue_count}")
    return payload


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "fetch-telegram":
        asyncio.run(_run_fetch_telegram(args))
    elif args.command == "export-structured-data":
        _run_export_structured_data(args)
    elif args.command == "export-facts":
        _run_export_facts(args)
    elif args.command == "data-audit":
        _run_data_audit(args)
    elif args.command == "eval-table-pipeline":
        _run_eval_table_pipeline(args)


if __name__ == "__main__":
    main()
