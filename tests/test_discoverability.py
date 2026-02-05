from paper_ocr.discoverability import (
    abstract_extraction_prompt,
    first_pages_excerpt,
    is_useful_discovery,
    normalize_discovery,
    render_group_readme,
    split_markdown_for_discovery,
)


def test_normalize_discovery_clamps_pages_and_topics():
    raw = {
        "paper_summary": "Paper about atomization.",
        "key_topics": ["sprays", "instability", ""],
        "sections": [
            {"title": "Intro", "start_page": 0, "end_page": 2, "summary": "Overview"},
            {"title": "Methods", "start_page": 3, "end_page": 999, "summary": "Model"},
        ],
    }
    out = normalize_discovery(raw, page_count=16)
    assert out["paper_summary"] == "Paper about atomization."
    assert out["key_topics"] == ["sprays", "instability"]
    assert out["sections"][0]["start_page"] == 1
    assert out["sections"][1]["end_page"] == 16


def test_render_group_readme_includes_locations_and_sections():
    text = render_group_readme(
        "lisa",
        [
            {
                "folder_name": "Madsen_Jesper_2007",
                "consolidated_markdown": "Computational and Experimental Study.md",
                "page_count": 200,
                "bibliography": {
                    "title": "Computational and Experimental Study of Sprays",
                    "citation": "Some Journal 1(2), 1-10. doi:10.1/abc",
                },
                "discovery": {
                    "paper_summary": "Compares experiments and CFD.",
                    "key_topics": ["sprays", "CFD"],
                    "sections": [
                        {
                            "title": "Method",
                            "start_page": 12,
                            "end_page": 45,
                            "summary": "Experimental setup and numerics.",
                        }
                    ],
                },
            }
        ],
    )
    assert "# Paper Index: lisa" in text
    assert "Madsen_Jesper_2007/Computational and Experimental Study.md" in text
    assert "Method (pp. 12-45)" in text


def test_split_markdown_for_discovery_covers_all_text():
    text = "A" * 35000 + "B" * 35000 + "C" * 1000
    chunks = split_markdown_for_discovery(text, max_chars=32000)
    assert len(chunks) >= 3
    assert "".join(chunks) == text


def test_is_useful_discovery_rejects_placeholder_values():
    bad = {"paper_summary": "string", "key_topics": ["string"], "sections": [{"title": "string"}]}
    good = {"paper_summary": "This paper analyzes instability in liquid sheets.", "key_topics": ["instability"], "sections": []}
    assert is_useful_discovery(bad) is False
    assert is_useful_discovery(good) is True


def test_first_pages_excerpt_limits_by_page_marker():
    md = (
        "# Page 1\nA\n\n# Page 2\nB\n\n# Page 3\nC\n\n# Page 4\nD\n\n# Page 5\nE\n\n# Page 6\nF\n"
    )
    out = first_pages_excerpt(md, max_pages=3)
    assert "# Page 1" in out
    assert "# Page 3" in out
    assert "# Page 4" not in out


def test_abstract_prompt_mentions_abstract_task():
    prompt = abstract_extraction_prompt("Title", "Citation", 10, "# Page 1\nAbstract text")
    assert "extract the paper abstract" in prompt.lower()
