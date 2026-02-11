import csv
import json
from pathlib import Path

import paper_ocr.doi_resolution as doi_resolution


class _Resp:
    def __init__(self, status: int, body: str = "", headers: dict[str, str] | None = None):
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_resolve_dois_autodetects_lowercase_columns_and_canonicalizes(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text(
        "canonical_id,doi,url_landing,title,authors_first_last,year,journal_or_repository\n"
        "1,,https://doi.org/10.1208/PT0802032,Rheological characterization,Shah,2007,AAPS\n"
    )
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        method = req.get_method()
        if "/agency" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {"status": "ok", "message": {"DOI": "10.1208/PT0802032", "agency": {"id": "crossref"}}}
                ),
            )
        if method == "HEAD" and "/works/" in url:
            return _Resp(200, body="")
        if "select=DOI" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {
                        "status": "ok",
                        "message": {
                            "DOI": "10.1208/PT0802032",
                            "title": ["Rheological characterization"],
                            "author": [{"family": "Shah"}],
                            "issued": {"date-parts": [[2007]]},
                            "type": "journal-article",
                            "container-title": ["AAPS"],
                        },
                    }
                ),
            )
        raise AssertionError(f"unexpected url {method} {url}")

    cfg = doi_resolution.DoiResolutionConfig(
        input_csv=in_csv,
        output_dir=out_dir,
        crossref_mailto="test@example.com",
        urlopen=_fake_urlopen,
    )
    summary = doi_resolution.resolve_dois(cfg)

    rows = _read_csv(Path(summary["resolved_csv"]))
    assert len(rows) == 1
    assert rows[0]["doi_status"] == "canonicalized_crossref"
    assert rows[0]["doi_canonical"] == "10.1208/pt0802032"
    fetch_rows = _read_csv(Path(summary["fetch_ready_csv"]))
    assert fetch_rows == [{"DOI": "10.1208/pt0802032"}]


def test_resolve_dois_non_crossref_uses_doi_resolver_fallback(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("doi\n10.5281/zenodo.13938532\n")
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        if "/agency" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {"status": "ok", "message": {"DOI": "10.5281/zenodo.13938532", "agency": {"id": "datacite"}}}
                ),
            )
        if "https://doi.org/" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {
                        "DOI": "10.5281/zenodo.13938532",
                        "title": "Dataset title",
                        "author": [{"family": "Sarma"}],
                        "issued": {"date-parts": [[2024]]},
                    }
                ),
            )
        raise AssertionError(f"unexpected url {url}")

    summary = doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fake_urlopen,
        )
    )
    row = _read_csv(Path(summary["resolved_csv"]))[0]
    assert row["doi_status"] == "canonicalized_non_crossref"
    assert row["doi_agency"] == "datacite"
    assert row["doi_canonical"] == "10.5281/zenodo.13938532"


def test_resolve_dois_search_scoring_accepts_clear_winner(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("title,first_author,year,journal\nBetter Viscosity Models,Shah,2020,AAPS\n")
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        method = req.get_method()
        if "query.bibliographic" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {
                        "status": "ok",
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1000/better",
                                    "title": ["Better Viscosity Models"],
                                    "author": [{"family": "Shah"}],
                                    "issued": {"date-parts": [[2020]]},
                                    "type": "journal-article",
                                    "container-title": ["AAPS"],
                                },
                                {
                                    "DOI": "10.1000/other",
                                    "title": ["Unrelated title"],
                                    "author": [{"family": "Doe"}],
                                    "issued": {"date-parts": [[2017]]},
                                    "type": "posted-content",
                                    "container-title": ["Nowhere"],
                                },
                            ]
                        },
                    }
                ),
            )
        if "/agency" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {"status": "ok", "message": {"DOI": "10.1000/better", "agency": {"id": "crossref"}}}
                ),
            )
        if method == "HEAD" and "/works/" in url:
            return _Resp(200, body="")
        if "select=DOI" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {
                        "status": "ok",
                        "message": {
                            "DOI": "10.1000/BETTER",
                            "title": ["Better Viscosity Models"],
                            "author": [{"family": "Shah"}],
                            "issued": {"date-parts": [[2020]]},
                            "type": "journal-article",
                            "container-title": ["AAPS"],
                        },
                    }
                ),
            )
        raise AssertionError(f"unexpected {method} {url}")

    summary = doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fake_urlopen,
        )
    )
    row = _read_csv(Path(summary["resolved_csv"]))[0]
    assert row["doi_status"] == "inferred_crossref"
    assert row["doi_canonical"] == "10.1000/better"


def test_resolve_dois_search_scoring_marks_ambiguous_needs_review(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("title,first_author,year\nPolymer rheology,Lee,2020\n")
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        if "query.bibliographic" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {
                        "status": "ok",
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1000/a",
                                    "title": ["Polymer rheology study"],
                                    "author": [{"family": "Lee"}],
                                    "issued": {"date-parts": [[2020]]},
                                    "type": "journal-article",
                                },
                                {
                                    "DOI": "10.1000/b",
                                    "title": ["Polymer rheology methods"],
                                    "author": [{"family": "Lee"}],
                                    "issued": {"date-parts": [[2020]]},
                                    "type": "journal-article",
                                },
                            ]
                        },
                    }
                ),
            )
        raise AssertionError(f"unexpected {url}")

    summary = doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fake_urlopen,
        )
    )
    row = _read_csv(Path(summary["resolved_csv"]))[0]
    assert row["doi_status"] == "needs_review"
    assert row["doi_canonical"] == ""
    fetch_rows = _read_csv(Path(summary["fetch_ready_csv"]))
    assert fetch_rows == []


def test_parse_landing_metadata_supports_highwire_dc_and_jsonld():
    html = """
    <html><head>
      <meta name="citation_title" content="Highwire Title">
      <meta name="citation_author" content="Jane Doe">
      <meta name="citation_publication_date" content="2021-07-01">
      <meta name="dc.title" content="DC Title">
      <meta name="dc.creator" content="DC Author">
      <meta name="dc.date" content="2019">
      <script type="application/ld+json">
        {"@type":"ScholarlyArticle","name":"JSONLD Title","author":[{"name":"Alex Smith"}],"datePublished":"2020-01-01","isPartOf":{"name":"Journal X"}}
      </script>
    </head><body></body></html>
    """
    parsed = doi_resolution.parse_landing_metadata(html)
    assert parsed["title"] == "Highwire Title"
    assert parsed["first_author"] == "Jane Doe"
    assert parsed["year"] == "2021"
    assert parsed["container_title"] == "Journal X"


def test_resolve_dois_cache_hit_avoids_network(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("doi\n10.1208/pt0802032\n")
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        method = req.get_method()
        if "/agency" in url:
            return _Resp(
                200,
                body=json.dumps(
                    {"status": "ok", "message": {"DOI": "10.1208/pt0802032", "agency": {"id": "crossref"}}}
                ),
            )
        if method == "HEAD":
            return _Resp(200, body="")
        if "select=DOI" in url:
            return _Resp(
                200,
                body=json.dumps({"status": "ok", "message": {"DOI": "10.1208/PT0802032"}}),
            )
        raise AssertionError(f"unexpected {method} {url}")

    doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fake_urlopen,
        )
    )

    def _fail_urlopen(req, timeout):  # noqa: ANN001,ARG001
        raise AssertionError("network should not be called")

    summary = doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fail_urlopen,
            use_cache=True,
            refresh_cache=False,
        )
    )
    row = _read_csv(Path(summary["resolved_csv"]))[0]
    assert row["doi_status"] == "canonicalized_crossref"


def test_resolve_dois_row_count_matches_input_and_summary(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("doi,url\n,,\n10.1000/abc,\n")
    out_dir = tmp_path / "reports" / "doi_resolution"

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        url = req.full_url
        method = req.get_method()
        if "/agency" in url:
            return _Resp(
                200,
                body=json.dumps({"status": "ok", "message": {"DOI": "10.1000/abc", "agency": {"id": "crossref"}}}),
            )
        if method == "HEAD":
            return _Resp(200, body="")
        if "select=DOI" in url:
            return _Resp(200, body=json.dumps({"status": "ok", "message": {"DOI": "10.1000/ABC"}}))
        raise AssertionError(f"unexpected {method} {url}")

    summary = doi_resolution.resolve_dois(
        doi_resolution.DoiResolutionConfig(
            input_csv=in_csv,
            output_dir=out_dir,
            urlopen=_fake_urlopen,
        )
    )
    rows = _read_csv(Path(summary["resolved_csv"]))
    assert len(rows) == 2
    payload = json.loads(Path(summary["summary_json"]).read_text())
    assert payload["total_rows"] == 2
