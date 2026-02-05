from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")


def discover_pdfs(in_dir: Path) -> list[Path]:
    return sorted([p for p in in_dir.rglob("*.pdf") if p.is_file()])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_id_from_sha(sha: str) -> str:
    return sha[:12]


def output_dir_name(pdf_path: Path) -> str:
    safe = _safe_name(pdf_path.stem)
    return safe or doc_id_from_sha(file_sha256(pdf_path))


def output_group_name(pdf_path: Path) -> str:
    safe = _safe_name(pdf_path.parent.name)
    return safe or "root"
