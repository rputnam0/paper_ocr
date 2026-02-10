from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split() if t]


@dataclass
class TextHeuristics:
    char_count: int
    printable_ratio: float
    cid_ratio: float
    replacement_char_ratio: float
    avg_token_length: float


@dataclass
class RouteDecision:
    route: str
    heuristics: TextHeuristics


def compute_text_heuristics(page_dict: dict[str, Any]) -> TextHeuristics:
    text_chunks: list[str] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if t:
                        text_chunks.append(t)
    text = "\n".join(text_chunks)
    stripped = "".join(ch for ch in text if not ch.isspace())
    char_count = len(stripped)
    tokens = _tokenize(text)

    # Compute printable ratio on non-whitespace characters only.
    # Newline/tab separators are expected in text extraction and should not
    # downgrade otherwise clean born-digital pages.
    non_whitespace = [ch for ch in text if not ch.isspace()]
    printable = sum(1 for ch in non_whitespace if ch.isprintable())
    total = max(len(non_whitespace), 1)
    printable_ratio = printable / total

    cid_hits = sum(1 for t in tokens if "cid:" in t.lower())
    cid_ratio = cid_hits / max(len(tokens), 1)

    replacement_char_ratio = text.count("�") / max(len(text), 1)

    avg_token_length = sum(len(t) for t in tokens) / max(len(tokens), 1)

    return TextHeuristics(
        char_count=char_count,
        printable_ratio=printable_ratio,
        cid_ratio=cid_ratio,
        replacement_char_ratio=replacement_char_ratio,
        avg_token_length=avg_token_length,
    )


def decide_route(heuristics: TextHeuristics, mode: str = "auto") -> str:
    if mode in {"anchored", "unanchored"}:
        return mode

    if (
        heuristics.char_count >= 200
        and heuristics.printable_ratio >= 0.85
        and heuristics.cid_ratio <= 0.02
    ):
        return "anchored"
    return "unanchored"


def is_text_only_candidate(heuristics: TextHeuristics) -> bool:
    return (
        heuristics.char_count >= 500
        and heuristics.printable_ratio >= 0.9
        and heuristics.cid_ratio <= 0.01
        and heuristics.replacement_char_ratio <= 0.001
    )


def is_structured_page_candidate(heuristics: TextHeuristics) -> bool:
    # Broader born-digital gate for layout-rich pages (tables/equations) where
    # text density may be moderate but encoding quality is still clean.
    if heuristics.printable_ratio < 0.85:
        return False
    if heuristics.cid_ratio > 0.05:
        return False
    if heuristics.replacement_char_ratio > 0.01:
        return False
    return heuristics.char_count >= 80
