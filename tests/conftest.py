from __future__ import annotations

import os

import pytest

from paper_ocr.test_selection import GATED_MARKERS, auto_markers_for_nodeid, enabled_markers


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run tests marked slow.")
    parser.addoption("--run-integration", action="store_true", default=False, help="Run integration tests.")
    parser.addoption("--run-network", action="store_true", default=False, help="Run tests that call network resources.")
    parser.addoption("--run-service", action="store_true", default=False, help="Run tests that require external services.")
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run tests that require GPU resources.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running tests")
    config.addinivalue_line("markers", "integration: cross-module integration tests")
    config.addinivalue_line("markers", "network: tests requiring outbound network access")
    config.addinivalue_line("markers", "service: tests requiring running external services")
    config.addinivalue_line("markers", "gpu: tests requiring GPU resources")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    options = {f"run_{m}": bool(config.getoption(f"--run-{m}")) for m in GATED_MARKERS}
    enabled = enabled_markers(options=options, env=os.environ)

    for item in items:
        for marker in auto_markers_for_nodeid(item.nodeid):
            item.add_marker(getattr(pytest.mark, marker))

        for marker in GATED_MARKERS:
            if item.get_closest_marker(marker) and marker not in enabled:
                reason = (
                    f"skipped '{marker}' test in fast lane; "
                    f"enable with --run-{marker} or PAPER_OCR_TEST_FULL=1"
                )
                item.add_marker(pytest.mark.skip(reason=reason))
                break

