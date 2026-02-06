from __future__ import annotations

import argparse
import asyncio
import json
import os
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

METADATA_MODEL_DEFAULT = "nvidia/Nemotron-3-Nano-30B-A3B"


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
    run.add_argument("--text-only", action="store_true", help="Enable text-only extraction for high-quality text layers")
    run.add_argument("--metadata-model", type=str, default=METADATA_MODEL_DEFAULT)

    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPINFRA_API_KEY. Set it in .env or environment.")
    return api_key


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
    return group_dir / author_year


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

    first_page["output_files"] = output_files
    return first_page


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
) -> dict[str, Any]:
    page = doc.load_page(page_index)
    page_dict = page.get_text("dict")
    heuristics = compute_text_heuristics(page_dict)
    route = decide_route(heuristics, mode=mode)

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
        write_text(page_path, text)
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

    write_text(page_path, parsed.markdown)
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

        pages: list[dict[str, Any]] = []
        if page_count > 0:
            first_page = await _process_page(
                semaphore=semaphore,
                client=client,
                doc=doc,
                page_index=0,
                mode=args.mode,
                max_tokens=args.max_tokens,
                model=args.model,
                scan_preprocess=args.scan_preprocess,
                debug=args.debug,
                dirs=dirs,
                force=args.force,
                text_only_enabled=args.text_only,
            )
            first_page_md = Path(first_page["output_files"].get("markdown", ""))
            first_page_text = first_page_md.read_text() if first_page_md.exists() else ""
            bibliography = await _extract_bibliography(client, args.metadata_model, first_page_text)
            doc_dir = _final_doc_dir(args, pdf_path, bibliography)
            final_dirs = ensure_dirs(doc_dir)
            first_page = _move_first_page_artifacts(first_page, dirs, final_dirs, args.debug)
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

        tasks = [
            _process_page(
                semaphore=semaphore,
                client=client,
                doc=doc,
                page_index=i,
                mode=args.mode,
                max_tokens=args.max_tokens,
                model=args.model,
                scan_preprocess=args.scan_preprocess,
                debug=args.debug,
                dirs=final_dirs,
                force=args.force,
                text_only_enabled=args.text_only,
            )
            for i in range(1, page_count)
        ]

        if tasks:
            pages.extend(await asyncio.gather(*tasks))
        pages_sorted = sorted(pages, key=lambda p: p["page_index"])
        for p in pages_sorted:
            manifest["pages"].append(p)
        manifest["bibliography"] = bibliography

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
        write_json(final_dirs["metadata"] / "discovery.json", discovery)
        write_json(final_dirs["metadata"] / "sections.json", discovery.get("sections", []))
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


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if args.command == "run":
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
