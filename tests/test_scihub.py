from pathlib import Path

from paper_ocr import scihub


def test_parse_scihub_base_urls_normalizes_and_deduplicates():
    got = scihub.parse_scihub_base_urls(
        "sci-hub.ru, https://sci-hub.ru/, http://sci-hub.st https://sci-hub.st"
    )
    assert got == ["https://sci-hub.ru", "http://sci-hub.st", "https://sci-hub.st"]


def test_discover_scihub_base_urls_filters_noise(monkeypatch):
    html = b"""
    <a href="https://sci-hub.ru">ru</a>
    <a href="https://sci-hub.ru/amp">amp</a>
    <a href="https://sci-hub.st">st</a>
    <a href="https://how-to-use-sci-hub">noise</a>
    <a href="https://example.com/sci-hub.ru">noise2</a>
    """

    monkeypatch.setattr(scihub, "_http_get", lambda url, timeout: (html, "text/html"))

    got = scihub.discover_scihub_base_urls(timeout=3)
    assert got == ["https://sci-hub.ru", "https://sci-hub.st"]


def test_resolve_pdf_url_uses_object_or_download_without_iframe(monkeypatch):
    html = b"""
    <html><body>
    <object type="application/pdf" data="/storage/2024/abc/paper.pdf#navpanes=0"></object>
    <a href="/download/2024/abc/paper.pdf">download</a>
    </body></html>
    """

    monkeypatch.setattr(scihub, "_http_get", lambda url, timeout: (html, "text/html"))

    got = scihub._resolve_pdf_url("10.1016/j.carbpol.2020.117012", "https://sci-hub.ru", timeout=4)
    assert got == "https://sci-hub.ru/storage/2024/abc/paper.pdf"


def test_download_pdf_via_scihub_works_with_download_anchor(monkeypatch, tmp_path: Path):
    pdf_bytes = b"%PDF-1.7\\nabc"
    lookup_html = b'<a href="/download/2024/abc/paper.pdf">download</a>'

    def _fake_http_get(url: str, timeout: int):
        if url.startswith("https://sci-hub.ru/10.1000/abc"):
            return lookup_html, "text/html"
        if url.startswith("https://sci-hub.ru/download/2024/abc/paper.pdf"):
            return pdf_bytes, "application/pdf"
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(scihub, "_http_get", _fake_http_get)
    out = tmp_path / "paper.pdf"

    got = scihub.download_pdf_via_scihub(
        identifier="10.1000/abc",
        output_path=out,
        timeout=4,
        base_urls=["https://sci-hub.ru"],
    )

    assert got == out
    assert out.read_bytes() == pdf_bytes

