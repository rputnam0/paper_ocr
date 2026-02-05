from pathlib import Path

from paper_ocr.ingest import output_dir_name, output_group_name


def test_output_dir_name_sanitizes():
    path = Path("/tmp/Weird Name (v1).pdf")
    assert output_dir_name(path) == "Weird_Name_v1"


def test_output_group_name_uses_pdf_parent():
    path = Path("/tmp/Carreau Yasuda/Weird Name (v1).pdf")
    assert output_group_name(path) == "Carreau_Yasuda"
