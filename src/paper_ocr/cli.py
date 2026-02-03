from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .anchoring import build_anchored_prompt, build_unanchored_prompt, extract_anchors
from .client import call_olmocr
from .ingest import discover_pdfs, doc_id_from_sha, file_sha256, output_dir_name
from .inspect import compute_text_heuristics, decide_route, is_text_only_candidate
from .postprocess import parse_yaml_front_matter
from .render import render_page
from .schemas import new_manifest
from .store import ensure_dirs, write_json, write_text


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


async def _process_pdf(args: argparse.Namespace, pdf_path: Path) -> None:
    sha = file_sha256(pdf_path)
    doc_id = doc_id_from_sha(sha)
    doc_dir = args.out_dir / output_dir_name(pdf_path)
    dirs = ensure_dirs(doc_dir)

    with fitz.open(pdf_path) as doc:
        manifest = new_manifest(
            doc_id=doc_id,
            source_path=str(pdf_path),
            sha256=sha,
            page_count=doc.page_count,
            model=args.model,
            base_url=args.base_url,
            prompt_version="v1",
        )

        api_key = _require_api_key()
        client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
        semaphore = asyncio.Semaphore(args.workers)

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
                dirs=dirs,
                force=args.force,
                text_only_enabled=args.text_only,
            )
            for i in range(doc.page_count)
        ]

        pages = await asyncio.gather(*tasks)
        pages_sorted = sorted(pages, key=lambda p: p["page_index"])
        for p in pages_sorted:
            manifest["pages"].append(p)

        write_json(dirs["metadata"] / "manifest.json", manifest)

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

        consolidated_name = f"{pdf_path.stem}.md"
        write_text(doc_dir / consolidated_name, "".join(md_out).strip() + "\n")
        write_text(dirs["metadata"] / "document.jsonl", "\n".join(jsonl_out) + "\n")


async def _run(args: argparse.Namespace) -> None:
    pdfs = discover_pdfs(args.in_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        await _process_pdf(args, pdf)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if args.command == "run":
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
