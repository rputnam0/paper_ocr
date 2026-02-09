from pathlib import Path

from paper_ocr.ingest import discover_pdfs, output_dir_name, output_group_name


def test_output_dir_name_sanitizes():
    path = Path("/tmp/Weird Name (v1).pdf")
    assert output_dir_name(path) == "Weird_Name_v1"


def test_output_group_name_uses_pdf_parent():
    path = Path("/tmp/Carreau Yasuda/Weird Name (v1).pdf")
    assert output_group_name(path) == "Carreau_Yasuda"


def test_discover_pdfs_is_case_insensitive(tmp_path: Path):
    pdf_lower = tmp_path / "a.pdf"
    pdf_upper = tmp_path / "b.PDF"
    txt = tmp_path / "c.txt"
    pdf_lower.write_bytes(b"pdf")
    pdf_upper.write_bytes(b"pdf")
    txt.write_text("x")
    discovered = discover_pdfs(tmp_path)
    assert pdf_lower in discovered
    assert pdf_upper in discovered
    assert txt not in discovered
