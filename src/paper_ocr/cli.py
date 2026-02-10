from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .anchoring import build_anchored_prompt, build_unanchored_prompt, extract_anchors
from .bibliography import (
    bibliography_prompt,
    citation_from_bibliography,
    extract_json_object,
    folder_name_from_bibliography,
    markdown_filename_from_title,
    normalize_bibliography,
)
from .client import call_olmocr, call_text_model
from .discoverability import (
    abstract_extraction_prompt,
    first_pages_excerpt,
    is_useful_discovery,
    normalize_discovery,
    render_group_readme,
)
from .ingest import discover_pdfs, doc_id_from_sha, file_sha256, output_dir_name, output_group_name
from .inspect import compute_text_heuristics, decide_route, is_text_only_candidate
from .postprocess import parse_yaml_front_matter
from .render import render_page
from .schemas import new_manifest
from .store import ensure_dirs, write_json, write_text
from .structured_data import build_structured_exports
from .structured_extract import (
    is_structured_candidate_doc,
    normalize_markdown_for_llm,
    run_grobid_doc,
    run_marker_page,
)
from .telegram_fetch import FetchTelegramConfig, fetch_from_telegram

METADATA_MODEL_DEFAULT = "nvidia/Nemotron-3-Nano-30B-A3B"
MIN_DELAY_DEFAULT = "4"
MAX_DELAY_DEFAULT = "8"
RESPONSE_TIMEOUT_DEFAULT = 15
SEARCH_TIMEOUT_DEFAULT = 40
CSV_JOB_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
DIGITAL_STRUCTURED_DEFAULT = "auto"
MARKER_COMMAND_DEFAULT = "marker_single"
MARKER_URL_DEFAULT = ""
MARKER_TIMEOUT_DEFAULT = "120"
GROBID_TIMEOUT_DEFAULT = "60"
DEPLOT_TIMEOUT_DEFAULT = "90"


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

    fetch = sub.add_parser("fetch-telegram", help="Fetch PDFs from Telegram bot using DOI CSV")
    fetch.add_argument("doi_csv", type=Path)
    fetch.add_argument("output_root", type=Path, nargs="?", default=Path("data/telegram_jobs"))
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
    author_year = folder_name_from_bibliography(normalize_bibliography(bibliography))
    if author_year in {"unknown_paper", "unknown_author_unknown_year"}:
        author_year = output_dir_name(pdf_path)
    candidate = group_dir / author_year
    current_sha = file_sha256(pdf_path)
    manifest_path = candidate / "metadata" / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            existing_sha = str(existing.get("sha256", "")).strip()
            if existing_sha and existing_sha != current_sha:
                return group_dir / f"{author_year}_{doc_id_from_sha(current_sha)}"
        except Exception:
            pass
    elif candidate.exists():
        # If folder exists without a manifest (e.g. interrupted prior run), avoid
        # merging unrelated PDFs into a shared author/year path.
        try:
            has_files = any(candidate.iterdir())
        except Exception:
            has_files = True
        if has_files:
            return group_dir / f"{author_year}_{doc_id_from_sha(current_sha)}"
    return candidate


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

    abstract_schema = (
        '{'
        '"abstract":"...",'
        '"key_topics":["..."],'
        '}'
    )
    excerpt = first_pages_excerpt(consolidated_markdown, max_pages=5)
    abstract_prompt = abstract_extraction_prompt(
        title=str(bibliography.get("title", "")),
        citation=str(bibliography.get("citation", "")),
        page_count=page_count,
        first_pages_markdown=excerpt,
    )
    abstract_raw = await _json_call(abstract_prompt, max_tokens=700, schema=abstract_schema)
    if abstract_raw:
        discovery = normalize_discovery(
            {
                "paper_summary": str(abstract_raw.get("abstract", "")).strip(),
                "key_topics": abstract_raw.get("key_topics", []),
                "sections": [],
            },
            page_count=page_count,
        )
        if is_useful_discovery(discovery):
            return discovery
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
                    bibliography = _merge_bibliography_with_patch(bibliography, grobid_result.bibliography_patch)
                else:
                    grobid_error = grobid_result.error
            doc_dir = _final_doc_dir(args, pdf_path, bibliography)
            final_dirs = ensure_dirs(doc_dir)
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
            doc_dir = _final_doc_dir(args, pdf_path, bibliography)
            final_dirs = ensure_dirs(doc_dir)

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
        }
        if grobid_error:
            structured_manifest["grobid_error"] = grobid_error
        manifest["structured_extraction"] = structured_manifest

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

        consolidated_name = markdown_filename_from_title(bibliography.get("title", ""))
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
        }
        if structured_data_enabled:
            try:
                summary = build_structured_exports(
                    doc_dir=doc_dir,
                    deplot_command=deplot_command,
                    deplot_timeout=deplot_timeout,
                )
                structured_data_manifest.update(
                    {
                        "table_count": summary.table_count,
                        "figure_count": summary.figure_count,
                        "deplot_count": summary.deplot_count,
                        "unresolved_figure_count": summary.unresolved_figure_count,
                        "errors": summary.errors,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                structured_data_manifest["errors"] = [str(exc)]
        manifest["structured_data_extraction"] = structured_data_manifest
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
    pdfs = discover_pdfs(args.in_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for pdf in pdfs:
        records.append(await _process_pdf(args, pdf))
    _write_group_readmes(records)


async def _run_fetch_telegram(args: argparse.Namespace) -> None:
    if not args.doi_csv.exists():
        raise SystemExit(f"DOI CSV does not exist: {args.doi_csv}")
    if not str(args.target_bot).strip():
        raise SystemExit("Missing target bot. Set TARGET_BOT in .env or pass --target-bot.")

    api_id, api_hash = _require_telegram_credentials()
    csv_name = CSV_JOB_SAFE_RE.sub("_", args.doi_csv.stem).strip("_") or "job"
    job_dir = args.output_root / csv_name
    input_dir = job_dir / "input"
    pdf_dir = job_dir / "pdfs"
    reports_dir = job_dir / "reports"
    ocr_out_dir = job_dir / "ocr_out"
    input_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ocr_out_dir.mkdir(parents=True, exist_ok=True)
    copied_csv = input_dir / args.doi_csv.name
    if args.doi_csv.resolve() != copied_csv.resolve():
        shutil.copy2(args.doi_csv, copied_csv)

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

    for doc_dir in docs:
        metadata_dir = doc_dir / "metadata"
        manifest_path = metadata_dir / "manifest.json"
        try:
            summary = build_structured_exports(
                doc_dir=doc_dir,
                deplot_command=str(getattr(args, "deplot_command", "") or ""),
                deplot_timeout=int(getattr(args, "deplot_timeout", int(DEPLOT_TIMEOUT_DEFAULT))),
            )
            structured_data_manifest: dict[str, Any] = {
                "enabled": True,
                "table_count": summary.table_count,
                "figure_count": summary.figure_count,
                "deplot_count": summary.deplot_count,
                "unresolved_figure_count": summary.unresolved_figure_count,
                "errors": summary.errors,
            }
        except Exception as exc:  # noqa: BLE001
            structured_data_manifest = {
                "enabled": True,
                "table_count": 0,
                "figure_count": 0,
                "deplot_count": 0,
                "unresolved_figure_count": 0,
                "errors": [str(exc)],
            }

        manifest: dict[str, Any]
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text())
                manifest = loaded if isinstance(loaded, dict) else {}
            except Exception:
                manifest = {}
        else:
            manifest = {}
        manifest["structured_data_extraction"] = structured_data_manifest
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
    return totals


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
    if args.command == "fetch-telegram":
        asyncio.run(_run_fetch_telegram(args))
    if args.command == "export-structured-data":
        _run_export_structured_data(args)


if __name__ == "__main__":
    main()
