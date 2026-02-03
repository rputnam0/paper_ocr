from pathlib import Path

from paper_ocr.ingest import output_dir_name


def test_output_dir_name_sanitizes():
    path = Path("/tmp/Weird Name (v1).pdf")
    assert output_dir_name(path) == "Weird_Name_v1"
