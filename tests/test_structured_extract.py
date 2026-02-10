from __future__ import annotations

import base64
import json
from pathlib import Path

import fitz

from paper_ocr.inspect import TextHeuristics
from paper_ocr.structured_extract import (
    is_structured_candidate_doc,
    normalize_markdown_for_llm,
    run_grobid_doc,
    run_marker_page,
)


def _h(char_count: int, printable: float = 0.99, cid: float = 0.0, repl: float = 0.0) -> TextHeuristics:
    return TextHeuristics(
        char_count=char_count,
        printable_ratio=printable,
        cid_ratio=cid,
        replacement_char_ratio=repl,
        avg_token_length=5.0,
    )


def _make_pdf(path: Path) -> None:
    with fitz.open() as doc:
        doc.new_page()
        doc.save(path)


def test_is_structured_candidate_doc_modes():
    routes = ["anchored"] * 10
    heuristics = [_h(800) for _ in range(10)]
    assert is_structured_candidate_doc("on", routes, heuristics)
    assert not is_structured_candidate_doc("off", routes, heuristics)
    assert is_structured_candidate_doc("auto", routes, heuristics)


def test_is_structured_candidate_doc_auto_thresholds():
    routes = ["anchored"] * 7 + ["unanchored"] * 3
    heuristics = [_h(800) for _ in range(6)] + [_h(100)] * 4
    assert is_structured_candidate_doc("auto", routes, heuristics)


def test_is_structured_candidate_doc_allows_image_heavy_cover_when_body_is_strong():
    routes = ["unanchored"] + ["anchored"] * 7 + ["unanchored"] * 2
    heuristics = [_h(100)] + [_h(800) for _ in range(6)] + [_h(100) for _ in range(3)]
    assert is_structured_candidate_doc("auto", routes, heuristics)


def test_is_structured_candidate_doc_rejects_weak_body_when_first_page_is_unanchored():
    routes = ["unanchored"] + ["anchored"] * 6 + ["unanchored"] * 3
    heuristics = [_h(100)] + [_h(800) for _ in range(6)] + [_h(100) for _ in range(3)]
    assert not is_structured_candidate_doc("auto", routes, heuristics)


def test_normalize_markdown_for_llm():
    inp = "##Intro\n\n-  item\n\nFigure 1:test\n\n|a|b|\n|---|---|\n|1|2|"
    out = normalize_markdown_for_llm(inp)
    assert "## Intro" in out
    assert "- item" in out
    assert "Figure 1: test" in out
    assert "| a | b |" in out


def test_run_marker_page_sets_ocr_engine_none(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)
    assets_root = tmp_path / "assets"
    seen_env = {}
    seen_cmd: list[str] = []

    def _fake_run(cmd, check, env, stdout, stderr, timeout):  # noqa: ANN001
        seen_env.update(env)
        seen_cmd[:] = cmd
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "doc.md").write_text("# Title\n")
        (out_dir / "doc.json").write_text(json.dumps({"ok": True}))
        return 0

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = run_marker_page(
        pdf_path=pdf_path,
        page_index=0,
        marker_command="marker_single",
        timeout=10,
        assets_root=assets_root,
        asset_level="standard",
    )
    assert result.success
    assert seen_env.get("OCR_ENGINE") == "None"
    assert "--disable_ocr" in seen_cmd
    assert "page_0001" in result.artifacts["markdown"]
    assert "page_0001" in result.artifacts["json"]


def test_run_marker_page_does_not_duplicate_disable_ocr(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)
    seen_cmd: list[str] = []

    def _fake_run(cmd, check, env, stdout, stderr, timeout):  # noqa: ANN001
        seen_cmd[:] = cmd
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "doc.md").write_text("# Title\n")
        return 0

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = run_marker_page(
        pdf_path=pdf_path,
        page_index=0,
        marker_command="marker_single --disable_ocr",
        timeout=10,
        assets_root=tmp_path / "assets",
        asset_level="standard",
    )
    assert result.success
    assert seen_cmd.count("--disable_ocr") == 1


def test_run_marker_page_failure_when_no_markdown(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    def _fake_run(cmd, check, env, stdout, stderr, timeout):  # noqa: ANN001,ARG001
        if "--output_dir" in cmd:
            out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        else:
            out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr("subprocess.run", _fake_run)
    result = run_marker_page(
        pdf_path=pdf_path,
        page_index=0,
        marker_command="marker_single",
        timeout=10,
        assets_root=tmp_path / "assets",
        asset_level="standard",
    )
    assert not result.success
    assert "markdown" in result.error.lower()


def test_run_marker_page_via_service_success(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            payload = {
                "success": True,
                "output": "# Service Title\n\n![](service_fig_1.jpeg)\n",
                "images": {
                    "service_fig_1.jpeg": base64.b64encode(b"imgdata").decode(),
                },
                "metadata": {"source": "service"},
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should_not_call_subprocess")))

    result = run_marker_page(
        pdf_path=pdf_path,
        page_index=0,
        marker_command="marker_single",
        timeout=10,
        assets_root=tmp_path / "assets",
        asset_level="full",
        marker_url="http://127.0.0.1:8008",
    )
    assert result.success
    assert "Service Title" in result.markdown
    assert "page_0001.md" in result.artifacts["markdown"]
    assert "page_0001.json" in result.artifacts["json"]
    assert "assets_dir" in result.artifacts


def test_run_marker_page_via_service_failure(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return b'{"success": false, "error": "service failed"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    result = run_marker_page(
        pdf_path=pdf_path,
        page_index=0,
        marker_command="marker_single",
        timeout=10,
        assets_root=tmp_path / "assets",
        asset_level="standard",
        marker_url="http://127.0.0.1:8008",
    )
    assert not result.success
    assert "service failed" in result.error


def test_run_grobid_doc_success(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)
    tei = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Paper Title</title>
        <author><persName><forename>Jane</forename><surname>Doe</surname></persName></author>
      </titleStmt>
      <publicationStmt><date when=\"2024-01-01\"/></publicationStmt>
    </fileDesc>
  </teiHeader>
  <text><body><div><head>Introduction</head></div></body></text>
</TEI>
"""

    class _Resp:
        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return tei

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out_tei = tmp_path / "assets" / "structured" / "grobid" / "fulltext.tei.xml"
    result = run_grobid_doc(
        pdf_path=pdf_path,
        grobid_url="http://localhost:8070",
        timeout=10,
        tei_out_path=out_tei,
    )
    assert result.success
    assert result.bibliography_patch["title"] == "Paper Title"
    assert result.bibliography_patch["authors"] == ["Doe, Jane"]
    assert result.bibliography_patch["year"] == "2024"
    assert result.sections[0]["title"] == "Introduction"
    assert out_tei.exists()


def test_run_grobid_doc_failure(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    def _fake_urlopen(req, timeout):  # noqa: ANN001,ARG001
        raise TimeoutError("boom")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    result = run_grobid_doc(
        pdf_path=pdf_path,
        grobid_url="http://localhost:8070",
        timeout=10,
        tei_out_path=tmp_path / "x.xml",
    )
    assert not result.success
    assert "boom" in result.error


def test_run_grobid_doc_extracts_figures_tables(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)
    tei = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">
  <text>
    <body>
      <figure type=\"table\" coords=\"2,10,20,30,40\">
        <label>Table 2</label>
        <figDesc>Viscosity values for polymers.</figDesc>
      </figure>
      <figure coords=\"3,11,21,31,41;3,15,25,35,45\">
        <label>Figure 5</label>
        <head>Flow curve</head>
      </figure>
    </body>
  </text>
</TEI>
"""

    class _Resp:
        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return tei

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    tei_out = tmp_path / "assets" / "structured" / "grobid" / "fulltext.tei.xml"
    result = run_grobid_doc(
        pdf_path=pdf_path,
        grobid_url="http://localhost:8070",
        timeout=10,
        tei_out_path=tei_out,
        doc_id="doc_abc123",
    )
    assert result.success
    assert len(result.figures_tables) == 2
    assert result.figures_tables[0]["doc_id"] == "doc_abc123"
    assert result.figures_tables[0]["type"] == "table"
    assert result.figures_tables[0]["label"] == "Table 2"
    assert result.figures_tables[0]["page"] == 2
    assert len(result.figures_tables[1]["coords"]) == 2
    assert "figures_tables" in result.artifacts
    figures_tables_path = Path(result.artifacts["figures_tables"])
    assert figures_tables_path.exists()
    lines = [ln for ln in figures_tables_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
