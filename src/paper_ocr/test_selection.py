from __future__ import annotations

from collections.abc import Mapping

GATED_MARKERS = ("slow", "integration", "network", "service", "gpu")

ENV_MAP = {
    "slow": "PAPER_OCR_RUN_SLOW",
    "integration": "PAPER_OCR_RUN_INTEGRATION",
    "network": "PAPER_OCR_RUN_NETWORK",
    "service": "PAPER_OCR_RUN_SERVICE",
    "gpu": "PAPER_OCR_RUN_GPU",
}

AUTO_INTEGRATION_NODEID_PREFIXES = (
    "tests/test_cli.py::",
    "tests/test_structured_extract.py::",
    "tests/test_structured_data.py::",
    "tests/test_eval_table_pipeline.py::",
)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def enabled_markers(options: Mapping[str, object], env: Mapping[str, str]) -> set[str]:
    if _truthy(env.get("PAPER_OCR_TEST_FULL")):
        return set(GATED_MARKERS)

    enabled: set[str] = set()
    for marker in GATED_MARKERS:
        option_name = f"run_{marker}"
        if bool(options.get(option_name)) or _truthy(env.get(ENV_MAP[marker])):
            enabled.add(marker)
    return enabled


def auto_markers_for_nodeid(nodeid: str) -> set[str]:
    if any(nodeid.startswith(prefix) for prefix in AUTO_INTEGRATION_NODEID_PREFIXES):
        return {"integration"}
    return set()

