from __future__ import annotations

from typing import Any


def discoverability_prompt(
    title: str,
    citation: str,
    page_count: int,
    markdown_excerpt: str,
) -> str:
    return (
        "You are extracting structured paper-discovery metadata from OCR markdown.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{"
        '"paper_summary": "string", '
        '"key_topics": ["string"], '
        '"sections": ['
        '{"title":"string","start_page":1,"end_page":1,"summary":"string"}'
        "]"
        "}\n"
        "Rules:\n"
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


def normalize_discovery(raw: dict[str, Any], page_count: int) -> dict[str, Any]:
    summary = str(raw.get("paper_summary", "")).strip()
    topics_raw = raw.get("key_topics", [])
    topics: list[str] = []
    if isinstance(topics_raw, list):
        for t in topics_raw:
            item = str(t).strip()
            if item:
                topics.append(item)

    sections_raw = raw.get("sections", [])
    sections: list[dict[str, Any]] = []
    if isinstance(sections_raw, list):
        for sec in sections_raw:
            if not isinstance(sec, dict):
                continue
            title = str(sec.get("title", "")).strip()
            if not title:
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
