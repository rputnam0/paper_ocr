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
        "render_contract": {},
        "runtime": {
            "pipeline_version": "v2-table-pipeline",
            "marker_version": "",
            "marker_config_hash": "",
            "grobid_version": "",
        },
        "table_pipeline": {
            "enabled": False,
            "marker_localized": False,
            "qa_mode": "warn",
            "qa_flags": 0,
        },
        "table_quality": {
            "empty_cell_ratio_threshold": 0.35,
            "repeated_text_ratio_threshold": 0.45,
            "column_instability_ratio_threshold": 0.20,
        },
        "table_qa": {
            "mode": "warn",
            "status": "not_run",
            "qa_skipped_reason": "",
        },
    }
