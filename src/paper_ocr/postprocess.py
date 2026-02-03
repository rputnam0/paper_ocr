from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class ParsedOutput:
    metadata: dict[str, Any]
    markdown: str


def parse_yaml_front_matter(text: str) -> ParsedOutput:
    if not text.startswith("---"):
        return ParsedOutput(metadata={}, markdown=text)

    parts = text.split("---", 2)
    if len(parts) < 3:
        return ParsedOutput(metadata={}, markdown=text)

    _, yaml_block, rest = parts
    try:
        metadata = yaml.safe_load(yaml_block) or {}
    except Exception:
        metadata = {}
        rest = text
    return ParsedOutput(metadata=metadata, markdown=rest.lstrip("\n"))
