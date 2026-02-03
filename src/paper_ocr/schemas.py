from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_manifest(
    doc_id: str,
    source_path: str,
    sha256: str,
    page_count: int,
    model: str,
    base_url: str,
    prompt_version: str,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_path": source_path,
        "sha256": sha256,
        "page_count": page_count,
        "model": model,
        "base_url": base_url,
        "prompt_version": prompt_version,
        "created_at": utc_now_iso(),
        "pages": [],
    }
