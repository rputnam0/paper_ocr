from __future__ import annotations

import json
import re
from typing import Any


def _is_placeholder_value(value: str) -> bool:
    norm = " ".join((value or "").strip().lower().split())
    if not norm:
        return True
    bad = {"string", "<summary text>", "<topic>", "<section title>", "<section summary>", "...", "…"}
    if norm in bad:
        return True
    if all(ch in ".-_ " for ch in norm):
        return True
    return False


def discoverability_prompt(
    title: str,
    citation: str,
    page_count: int,
    markdown_excerpt: str,
) -> str:
    return (
        "You are extracting structured paper-discovery metadata from OCR markdown.\n"
        "Respond with a SINGLE JSON object in assistant content only.\n"
        "Do not include analysis, markdown, code fences, or extra text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        '"paper_summary": "<summary text>", '
        '"key_topics": ["<topic>"], '
        '"sections": ['
        '{"title":"<section title>","start_page":1,"end_page":1,"summary":"<section summary>"}'
        "]"
        "}\n"
        "Rules:\n"
        "- Do NOT use placeholder words like 'string' or '<topic>' in output values.\n"
        "- Keep paper_summary to 1-3 sentences.\n"
        "- key_topics should be concise (3-8 entries).\n"
        "- sections should cover major content regions in reading order.\n"
        "- start_page/end_page are 1-based and between 1 and page_count.\n"
        "- If uncertain, still return best-effort fields; never include extra keys.\n\n"
        f"Title: {title or '(unknown)'}\n"
        f"Citation: {citation or '(unknown)'}\n"
        f"Page count: {page_count}\n\n"
        "OCR markdown excerpt:\n"
        f"{markdown_excerpt}"
    )


def abstract_extraction_prompt(
    title: str,
    citation: str,
    page_count: int,
    first_pages_markdown: str,
) -> str:
    return (
        "You extract the paper abstract from OCR markdown.\n"
        "Respond with a SINGLE JSON object in assistant content only.\n"
        "Do not include analysis, markdown, code fences, or extra text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        '"abstract":"<abstract text>",'
        '"key_topics":["<topic>"]'
        "}\n"
        "Rules:\n"
        "- Prefer the explicit abstract section when present.\n"
        "- If no explicit abstract exists, provide a brief high-level summary from the opening pages.\n"
        "- key_topics should be concise (3-8 entries).\n"
        "- Do NOT use placeholder values like 'string' or '...'.\n\n"
        f"Title: {title or '(unknown)'}\n"
        f"Citation: {citation or '(unknown)'}\n"
        f"Page count: {page_count}\n\n"
        "First pages markdown:\n"
        f"{first_pages_markdown}"
    )


def first_pages_excerpt(markdown_text: str, max_pages: int = 5) -> str:
    if max_pages < 1:
        max_pages = 1
    text = markdown_text or ""
    if not text.strip():
        return ""
    starts = list(re.finditer(r"(?m)^# Page (\d+)\s*$", text))
    if not starts:
        return text[:24000]
    if len(starts) <= max_pages:
        return text
    cutoff = starts[max_pages].start()
    return text[:cutoff]


def discoverability_chunk_prompt(
    title: str,
    citation: str,
    page_count: int,
    chunk_index: int,
    chunk_count: int,
    markdown_chunk: str,
) -> str:
    return (
        "You are extracting discovery metadata from one chunk of a paper markdown transcript.\n"
        "Respond with a SINGLE JSON object in assistant content only.\n"
        "Do not include analysis, markdown, code fences, or extra text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        '"chunk_summary": "<summary text>", '
        '"key_topics": ["<topic>"], '
        '"sections": ['
        '{"title":"<section title>","start_page":1,"end_page":1,"summary":"<section summary>"}'
        "]"
        "}\n"
        "Rules:\n"
        "- Do NOT use placeholder words like 'string' or '<topic>' in output values.\n"
        "- Infer absolute page numbers from `# Page N` markers in the chunk.\n"
        "- Keep chunk_summary to 1-2 sentences.\n"
        "- If uncertain, return best effort with empty fields as needed.\n\n"
        f"Title: {title or '(unknown)'}\n"
        f"Citation: {citation or '(unknown)'}\n"
        f"Page count: {page_count}\n"
        f"Chunk: {chunk_index}/{chunk_count}\n\n"
        "Markdown chunk:\n"
        f"{markdown_chunk}"
    )


def discoverability_aggregate_prompt(
    title: str,
    citation: str,
    page_count: int,
    chunk_outputs: list[dict[str, Any]],
) -> str:
    return (
        "You are combining chunk-level paper discovery data into one final structured output.\n"
        "Respond with a SINGLE JSON object in assistant content only.\n"
        "Do not include analysis, markdown, code fences, or extra text.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        '"paper_summary": "<summary text>", '
        '"key_topics": ["<topic>"], '
        '"sections": ['
        '{"title":"<section title>","start_page":1,"end_page":1,"summary":"<section summary>"}'
        "]"
        "}\n"
        "Rules:\n"
        "- Do NOT use placeholder words like 'string' or '<topic>' in output values.\n"
        "- paper_summary: 2-5 sentences high-level summary of the full paper.\n"
        "- key_topics: 3-10 concise topics.\n"
        "- sections: 4-12 major sections in reading order.\n"
        "- start_page/end_page are absolute 1-based pages.\n\n"
        f"Title: {title or '(unknown)'}\n"
        f"Citation: {citation or '(unknown)'}\n"
        f"Page count: {page_count}\n\n"
        "Chunk discovery JSON list:\n"
        f"{json.dumps(chunk_outputs, ensure_ascii=True)}"
    )


def split_markdown_for_discovery(markdown_text: str, max_chars: int = 32000) -> list[str]:
    if max_chars < 1000:
        max_chars = 1000
    text = markdown_text or ""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        if j < n:
            split = text.rfind("\n\n# Page ", i, j)
            if split > i:
                j = split
        if j <= i:
            j = min(i + max_chars, n)
        chunks.append(text[i:j])
        i = j
    return chunks


def normalize_discovery(raw: dict[str, Any], page_count: int) -> dict[str, Any]:
    summary = str(raw.get("paper_summary", "")).strip()
    if _is_placeholder_value(summary):
        summary = ""
    topics_raw = raw.get("key_topics", [])
    topics: list[str] = []
    if isinstance(topics_raw, list):
        for t in topics_raw:
            item = str(t).strip()
            if item and not _is_placeholder_value(item):
                topics.append(item)

    sections_raw = raw.get("sections", [])
    sections: list[dict[str, Any]] = []
    if isinstance(sections_raw, list):
        for sec in sections_raw:
            if not isinstance(sec, dict):
                continue
            title = str(sec.get("title", "")).strip()
            if not title or _is_placeholder_value(title):
                continue
            try:
                start_page = int(sec.get("start_page", 1))
            except Exception:
                start_page = 1
            try:
                end_page = int(sec.get("end_page", start_page))
            except Exception:
                end_page = start_page
            max_page = max(1, int(page_count or 1))
            start_page = max(1, min(start_page, max_page))
            end_page = max(start_page, min(end_page, max_page))
            sec_summary = str(sec.get("summary", "")).strip()
            if _is_placeholder_value(sec_summary):
                sec_summary = ""
            sections.append(
                {
                    "title": title,
                    "start_page": start_page,
                    "end_page": end_page,
                    "summary": sec_summary,
                }
            )

    return {
        "paper_summary": summary,
        "key_topics": topics,
        "sections": sections,
    }


def is_useful_discovery(discovery: dict[str, Any]) -> bool:
    summary = str(discovery.get("paper_summary", "")).strip()
    topics = [str(t).strip() for t in (discovery.get("key_topics", []) or [])]
    sections = discovery.get("sections", []) or []

    summary_ok = not _is_placeholder_value(summary) and len(summary) >= 20
    topics_ok = any(not _is_placeholder_value(t) for t in topics)
    sections_ok = any(
        not _is_placeholder_value(str((s or {}).get("title", "")))
        for s in sections
        if isinstance(s, dict)
    )
    return summary_ok or (topics_ok and sections_ok)


def render_group_readme(group_name: str, papers: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# Paper Index: {group_name}")
    lines.append("")
    lines.append("This index is generated for quick LLM discovery across papers in this folder.")
    lines.append("")
    for paper in sorted(papers, key=lambda p: p.get("folder_name", "")):
        bibliography = paper.get("bibliography", {}) or {}
        discovery = paper.get("discovery", {}) or {}
        folder = str(paper.get("folder_name", "")).strip()
        consolidated = str(paper.get("consolidated_markdown", "")).strip()
        title = str(bibliography.get("title", "")).strip() or consolidated.replace(".md", "") or folder
        citation = str(bibliography.get("citation", "")).strip()
        page_count = int(paper.get("page_count", 0) or 0)
        summary = str(discovery.get("paper_summary", "")).strip()
        topics = discovery.get("key_topics", []) or []
        sections = discovery.get("sections", []) or []

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Location: `{folder}/{consolidated}`")
        if page_count:
            lines.append(f"- Pages: {page_count}")
        if citation:
            lines.append(f"- Citation: {citation}")
        if summary:
            lines.append(f"- Summary: {summary}")
        if topics:
            lines.append(f"- Topics: {', '.join(str(t) for t in topics)}")
        if sections:
            lines.append("- Key Sections:")
            for sec in sections[:12]:
                sec_title = str(sec.get("title", "")).strip()
                if not sec_title:
                    continue
                start_page = int(sec.get("start_page", 1) or 1)
                end_page = int(sec.get("end_page", start_page) or start_page)
                sec_summary = str(sec.get("summary", "")).strip()
                if start_page == end_page:
                    page_span = f"p. {start_page}"
                else:
                    page_span = f"pp. {start_page}-{end_page}"
                if sec_summary:
                    lines.append(f"  - {sec_title} ({page_span}): {sec_summary}")
                else:
                    lines.append(f"  - {sec_title} ({page_span})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
