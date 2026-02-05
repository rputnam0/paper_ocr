from paper_ocr.store import ensure_dirs


def test_ensure_dirs_places_pages_at_doc_root(tmp_path):
    dirs = ensure_dirs(tmp_path / "doc")
    assert dirs["pages"].name == "pages"
    assert dirs["pages"].parent.name == "doc"
    assert dirs["metadata"].name == "metadata"
