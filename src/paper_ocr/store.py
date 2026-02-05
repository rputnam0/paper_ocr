from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dirs(doc_dir: Path) -> dict[str, Path]:
    metadata_dir = doc_dir / "metadata"
    pages_dir = doc_dir / "pages"
    assets_dir = metadata_dir / "assets"
    debug_dir = metadata_dir / "debug"
    pages_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metadata": metadata_dir,
        "pages": pages_dir,
        "assets": assets_dir,
        "debug": debug_dir,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
