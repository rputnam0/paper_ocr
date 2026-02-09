from __future__ import annotations

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

from .inspect import TextHeuristics, is_text_only_candidate


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
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""


def is_structured_candidate_doc(
    digital_structured: str,
    routes: list[str],
    heuristics_by_page: list[TextHeuristics],
) -> bool:
    mode = (digital_structured or "").strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    if mode != "auto":
        return False
    if not routes or not heuristics_by_page or len(routes) != len(heuristics_by_page):
        return False
    if routes[0] != "anchored":
        return False
    anchored_ratio = sum(1 for r in routes if r == "anchored") / len(routes)
    candidate_ratio = sum(1 for h in heuristics_by_page if is_text_only_candidate(h)) / len(heuristics_by_page)
    return anchored_ratio >= 0.7 and candidate_ratio >= 0.6


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
    return [
        [*base, str(input_pdf), "--output_dir", str(output_dir), "--output_format", "markdown"],
        [*base, str(input_pdf), "--output_dir", str(output_dir)],
        [*base, str(input_pdf), str(output_dir)],
    ]


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
) -> StructuredPageResult:
    env = os.environ.copy()
    env["OCR_ENGINE"] = "None"
    last_error = ""

    with tempfile.TemporaryDirectory(prefix="paper_ocr_marker_") as tmp:
        tmp_dir = Path(tmp)
        single_pdf = tmp_dir / f"page_{page_index + 1:04d}.pdf"
        _single_page_pdf(pdf_path, page_index, single_pdf)

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
    boundary = f"----paper-ocr-{uuid.uuid4().hex}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
    )
    body.extend(pdf_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request(url=url, method="POST", data=bytes(body))
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return req


def _tei_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _parse_grobid_tei(tei_xml: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    return patch, sections


def run_grobid_doc(
    pdf_path: Path,
    grobid_url: str,
    timeout: int,
    tei_out_path: Path,
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
        patch, sections = _parse_grobid_tei(tei_xml)
        return GrobidResult(
            success=True,
            tei_xml=tei_xml,
            bibliography_patch=patch,
            sections=sections,
            artifacts={"tei_xml": str(tei_out_path)},
        )
    except Exception as exc:  # noqa: BLE001
        return GrobidResult(success=False, error=str(exc))
