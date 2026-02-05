from paper_ocr.bibliography import (
    citation_from_bibliography,
    extract_json_object,
    folder_name_from_bibliography,
    markdown_filename_from_title,
    normalize_bibliography,
)


def test_extract_json_object_from_fenced_block():
    text = """Here you go:
```json
{"title":"T","authors":["Carreau, P. J."],"year":"1972"}
```
"""
    data = extract_json_object(text)
    assert data["title"] == "T"
    assert data["authors"] == ["Carreau, P. J."]
    assert data["year"] == "1972"


def test_folder_name_from_bibliography_author_year():
    info = normalize_bibliography(
        {
            "title": "Rheological Equations from Molecular Network Theories.",
            "authors": ["Carreau, P. J."],
            "year": "1972",
        }
    )
    assert folder_name_from_bibliography(info) == "Carreau_P_J_1972"


def test_markdown_filename_from_title_keeps_title_shape():
    title = "Rheological Equations from Molecular Network Theories."
    assert markdown_filename_from_title(title) == "Rheological Equations from Molecular Network Theories.md"


def test_citation_from_bibliography_uses_journal_ref_and_doi():
    info = normalize_bibliography(
        {
            "title": "T",
            "authors": ["Carreau, P. J."],
            "year": "1972",
            "journal_ref": "Transactions of the Society of Rheology 16(1), 99-127.",
            "doi": "10.1122/1.549276",
        }
    )
    assert (
        citation_from_bibliography(info)
        == "Transactions of the Society of Rheology 16(1), 99-127. doi:10.1122/1.549276"
    )
