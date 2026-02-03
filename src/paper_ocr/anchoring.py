from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import yaml


@dataclass
class AnchorBlock:
    text: str
    bbox: list[float]


@dataclass
class AnchorPayload:
    page_width: float
    page_height: float
    anchors: list[AnchorBlock]


def extract_anchors(page_dict: dict[str, Any], page_width: float, page_height: float) -> AnchorPayload:
    anchors: list[AnchorBlock] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = block.get("bbox", [0, 0, 0, 0])
        text_parts: list[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    text_parts.append(t)
        text = " ".join(text_parts).strip()
        if not text:
            continue
        x0, y0, x1, y1 = block_bbox
        if page_width <= 0 or page_height <= 0:
            continue
        norm = [x0 / page_width, y0 / page_height, x1 / page_width, y1 / page_height]
        anchors.append(AnchorBlock(text=text, bbox=norm))
    return AnchorPayload(page_width=page_width, page_height=page_height, anchors=anchors)


def _load_toolkit_prompt_fn() -> tuple[Callable[..., str] | None, str | None]:
    candidates = [
        ("olmocr.prompts", "build_no_anchoring_v4_yaml_prompt"),
        ("olmocr.prompting", "build_no_anchoring_v4_yaml_prompt"),
        ("olmocr", "build_no_anchoring_v4_yaml_prompt"),
    ]
    for module_name, fn_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, fn_name, None)
        if fn is not None:
            return fn, f"{module_name}.{fn_name}"
    return None, None


def build_unanchored_prompt(page_width: float, page_height: float) -> tuple[str, str]:
    toolkit_fn, builder_name = _load_toolkit_prompt_fn()
    if toolkit_fn is not None:
        try:
            prompt = toolkit_fn(page_width=page_width, page_height=page_height)
            return prompt, builder_name or "toolkit"
        except Exception:
            pass

    payload = {
        "mode": "unanchored",
        "page_width": page_width,
        "page_height": page_height,
        "instructions": "Extract all text in reading order. Preserve math as LaTeX.",
    }
    yaml_front = yaml.safe_dump(payload, sort_keys=False).strip()
    prompt = f"---\n{yaml_front}\n---\nReturn Markdown only."
    return prompt, "fallback"


def build_anchored_prompt(anchor_payload: AnchorPayload) -> tuple[str, str]:
    payload = {
        "mode": "anchored",
        "page_width": anchor_payload.page_width,
        "page_height": anchor_payload.page_height,
        "anchors": [
            {"text": a.text, "bbox": [round(v, 6) for v in a.bbox]} for a in anchor_payload.anchors
        ],
        "instructions": "Use anchors to resolve reading order and preserve math as LaTeX.",
    }
    yaml_front = yaml.safe_dump(payload, sort_keys=False).strip()
    prompt = f"---\n{yaml_front}\n---\nReturn Markdown only."
    return prompt, "fallback"
