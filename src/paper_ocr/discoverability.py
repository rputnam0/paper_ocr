from __future__ import annotations

import re
from typing import Any

PAGE_MARKER_RE = re.compile(r"(?m)^# Page (\d+)\s*$")
MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
PLAIN_SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|background|methods?|materials and methods|results?|discussion|conclusion|references)\s*$",
    flags=re.IGNORECASE,
)
ABSTRACT_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?abstract\s*:?\s*$", flags=re.IGNORECASE)
KEYWORDS_LINE_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?keywords?\s*:\s*(.+)$", flags=re.IGNORECASE)
TOPIC_STOPWORDS = {
    "about",
    "across",
    "after",
    "analysis",
    "approach",
    "based",
    "between",
    "conclusion",
    "conclusions",
    "data",
    "discussion",
    "from",
    "into",
    "introduction",
    "method",
    "methods",
    "paper",
    "result",
    "results",
    "study",
    "table",
    "tables",
    "using",
    "with",
}


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


def _clean_markdown_line(raw: str) -> str:
    line = str(raw or "").strip()
    if not line:
        return ""
    line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"[`*_~]+", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _split_markdown_pages(markdown_text: str) -> list[tuple[int, str]]:
    text = str(markdown_text or "")
    if not text.strip():
        return []
    markers = list(PAGE_MARKER_RE.finditer(text))
    if not markers:
        return [(1, text)]
    pages: list[tuple[int, str]] = []
    for idx, marker in enumerate(markers):
        page_num = int(marker.group(1))
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        pages.append((page_num, text[start:end]))
    return pages


def _clean_heading_text(raw: str) -> str:
    text = _clean_markdown_line(raw)
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text).strip()
    text = text.strip(":.- ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_heading_candidate(title: str) -> bool:
    text = str(title or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith("page "):
        return False
    if len(text) < 2 or len(text) > 120:
        return False
    if all(ch in "-_=*." for ch in text):
        return False
    return True


def _extract_sections_from_markdown(consolidated_markdown: str, page_count: int) -> list[dict[str, Any]]:
    pages = _split_markdown_pages(consolidated_markdown)
    if not pages:
        return []
    headings: list[dict[str, Any]] = []
    for page_num, page_text in pages:
        lines = page_text.splitlines()
        for idx, raw in enumerate(lines):
            line = str(raw or "").strip()
            if not line:
                continue
            title = ""
            heading_match = MD_HEADING_RE.match(line)
            if heading_match:
                title = _clean_heading_text(heading_match.group(2))
            elif PLAIN_SECTION_RE.match(line):
                title = _clean_heading_text(line)
            if not _is_heading_candidate(title):
                continue
            snippet = ""
            for follower in lines[idx + 1 :]:
                cleaned = _clean_markdown_line(follower)
                if not cleaned:
                    continue
                if MD_HEADING_RE.match(follower) or PLAIN_SECTION_RE.match(follower):
                    break
                if cleaned.lower().startswith("keywords"):
                    break
                snippet = cleaned
                break
            headings.append({"title": title, "page": page_num, "summary": snippet})
    # Deduplicate near-adjacent repeated headings.
    compact: list[dict[str, Any]] = []
    for heading in headings:
        title = str(heading.get("title", "")).strip()
        if compact and compact[-1]["title"].lower() == title.lower() and int(compact[-1]["page"]) == int(heading.get("page", 1)):
            continue
        compact.append(heading)

    max_page = max(1, int(page_count or 1))
    sections: list[dict[str, Any]] = []
    for idx, heading in enumerate(compact[:24]):
        start_page = max(1, min(int(heading.get("page", 1) or 1), max_page))
        if idx + 1 < len(compact):
            next_page = max(1, min(int(compact[idx + 1].get("page", start_page) or start_page), max_page))
            end_page = start_page if next_page <= start_page else max(start_page, next_page - 1)
        else:
            end_page = max(start_page, max_page)
        summary = str(heading.get("summary", "")).strip()
        if summary and len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        sections.append(
            {
                "title": str(heading.get("title", "")).strip(),
                "start_page": start_page,
                "end_page": end_page,
                "summary": summary,
            }
        )
    return sections


def _clip_sentences(text: str, max_sentences: int = 2, max_chars: int = 480) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+", raw)
    chosen = " ".join(pieces[:max_sentences]).strip() or raw
    if len(chosen) <= max_chars:
        return chosen
    return chosen[: max_chars - 3].rstrip() + "..."


def _extract_abstract(consolidated_markdown: str) -> str:
    excerpt = first_pages_excerpt(consolidated_markdown, max_pages=5)
    lines = excerpt.splitlines()
    collecting = False
    collected: list[str] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            if collecting and collected:
                break
            continue
        if ABSTRACT_HEADING_RE.match(line):
            collecting = True
            continue
        if collecting:
            if PAGE_MARKER_RE.match(line) or MD_HEADING_RE.match(line) or PLAIN_SECTION_RE.match(line):
                break
            if line.lower().startswith("keywords"):
                break
            cleaned = _clean_markdown_line(line)
            if cleaned:
                collected.append(cleaned)
    return _clip_sentences(" ".join(collected))


def _fallback_summary(consolidated_markdown: str) -> str:
    excerpt = first_pages_excerpt(consolidated_markdown, max_pages=2)
    chunks: list[str] = []
    for raw in excerpt.splitlines():
        line = str(raw or "").strip()
        if not line:
            if chunks:
                break
            continue
        if PAGE_MARKER_RE.match(line) or MD_HEADING_RE.match(line):
            continue
        cleaned = _clean_markdown_line(line)
        if cleaned and not cleaned.lower().startswith(("keywords", "doi:")):
            chunks.append(cleaned)
    return _clip_sentences(" ".join(chunks))


def _extract_keyword_topics(consolidated_markdown: str) -> list[str]:
    topics: list[str] = []
    for raw in consolidated_markdown.splitlines():
        match = KEYWORDS_LINE_RE.match(str(raw or ""))
        if not match:
            continue
        for part in re.split(r"[;,]", match.group(1)):
            candidate = str(part).strip(" .")
            if candidate and candidate.lower() not in {t.lower() for t in topics}:
                topics.append(candidate)
        if topics:
            break
    return topics[:8]


def _token_topics(text: str, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", str(text or "").lower()):
        if token in TOPIC_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def _derive_topics(consolidated_markdown: str, sections: list[dict[str, Any]], summary: str) -> list[str]:
    topics = _extract_keyword_topics(consolidated_markdown)
    if len(topics) < 3:
        section_blob = " ".join(str(sec.get("title", "")) for sec in sections)
        for topic in _token_topics(section_blob, limit=8):
            if topic.lower() not in {t.lower() for t in topics}:
                topics.append(topic)
            if len(topics) >= 8:
                break
    if len(topics) < 3:
        for topic in _token_topics(summary, limit=8):
            if topic.lower() not in {t.lower() for t in topics}:
                topics.append(topic)
            if len(topics) >= 8:
                break
    return topics[:8]


def extract_discovery_deterministic(
    consolidated_markdown: str,
    page_count: int,
) -> dict[str, Any]:
    if not str(consolidated_markdown or "").strip():
        return {"paper_summary": "", "key_topics": [], "sections": []}
    sections = _extract_sections_from_markdown(consolidated_markdown, page_count=page_count)
    summary = _extract_abstract(consolidated_markdown) or _fallback_summary(consolidated_markdown)
    topics = _derive_topics(consolidated_markdown, sections=sections, summary=summary)
    return normalize_discovery(
        {
            "paper_summary": summary,
            "key_topics": topics,
            "sections": sections,
        },
        page_count=page_count,
    )


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
