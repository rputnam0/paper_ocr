from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class ParsedOutput:
    metadata: dict[str, Any]
    markdown: str


def parse_yaml_front_matter(text: str) -> ParsedOutput:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return ParsedOutput(metadata={}, markdown=text)

    closing_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_idx = i
            break
    if closing_idx is None:
        return ParsedOutput(metadata={}, markdown=text)

    yaml_block = "".join(lines[1:closing_idx])
    rest = "".join(lines[closing_idx + 1 :])
    try:
        loaded = yaml.safe_load(yaml_block) or {}
        metadata = loaded if isinstance(loaded, dict) else {}
    except Exception:
        metadata = {}
        return ParsedOutput(metadata=metadata, markdown=text)
    return ParsedOutput(metadata=metadata, markdown=rest)
