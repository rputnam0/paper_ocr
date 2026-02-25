from paper_ocr.test_selection import (
    GATED_MARKERS,
    auto_markers_for_nodeid,
    enabled_markers,
)


def test_enabled_markers_default_is_fast_lane():
    got = enabled_markers(
        options={
            "run_slow": False,
            "run_integration": False,
            "run_network": False,
            "run_service": False,
            "run_gpu": False,
        },
        env={},
    )
    assert got == set()


def test_enabled_markers_full_mode_enables_all():
    got = enabled_markers(
        options={},
        env={"PAPER_OCR_TEST_FULL": "1"},
    )
    assert got == set(GATED_MARKERS)


def test_enabled_markers_respects_cli_flags():
    got = enabled_markers(
        options={
            "run_slow": True,
            "run_integration": True,
            "run_network": False,
            "run_service": False,
            "run_gpu": False,
        },
        env={},
    )
    assert got == {"slow", "integration"}


def test_auto_markers_for_nodeid_marks_heavy_modules_as_integration():
    assert auto_markers_for_nodeid("tests/test_cli.py::test_parse_fetch_telegram_defaults") == {"integration"}
    assert auto_markers_for_nodeid("tests/test_structured_extract.py::test_x") == {"integration"}
    assert auto_markers_for_nodeid("tests/test_bibliography.py::test_extract_json_object") == set()

